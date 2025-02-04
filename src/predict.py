import argparse
import pathlib
import re
import pandas as pd
from _version import __version__
from utils.logger import logger
from bullet import Bullet, Input, ScrollBar, SlidePrompt, Check, colors
from tqdm import tqdm
import torch
import spacy
import ast


parser = argparse.ArgumentParser(
    prog="sciPy SGS NER Predict",
    description="Utility for training a NER model",
)
parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
parser.add_argument("--data", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument(
    "--model", type=pathlib.Path, default=pathlib.Path("./release/model")
)
parser.add_argument("--sep", type=str)
parser.add_argument("--col", type=str)
parser.add_argument("--raw", choices=("True", "False"), default="True")
parser.add_argument("--out_col", type=str)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    prompt_data = get_prompt()
    logger.info("Iniciando o modelo...")
    logger.info(f"Device {device}")
    if device.type == "cuda":
        logger.info("Utilizando GPU para inferência.")
        spacy.prefer_gpu()

    model = spacy.load(args.model)

    df = prompt_data["dataframe"]
    df[prompt_data["column"]] = df[prompt_data["column"]].str.upper()
    df[prompt_data["column"]] = df[prompt_data["column"]].replace(
        r"\s+", " ", regex=True
    )
    df[prompt_data["column"]] = df[prompt_data["column"]].replace(
        r"LTDA.", "LTDA", regex=True
    )
    df[prompt_data["column"]] = df[prompt_data["column"]].str.strip()
    df.insert(
        df.columns.get_loc(prompt_data["column"]) + 1, prompt_data["output_column"], ""
    )
    df = df.sort_values(by=prompt_data["column"])
    df = df.reset_index(drop=True)

    ORGS_LIST = []
    ORGS_DOUBT = []

    logger.info(f"Realizando Processamento...")
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        result = None
        text = str(row[prompt_data["column"]])

        last_row_index = i - 1
        if last_row_index != -1:
            last_text = str(df.at[last_row_index, prompt_data["column"]])
            last_result = str(df.at[last_row_index, prompt_data["output_column"]])
            if last_result and (last_text and last_text == text):
                result = last_result

        if result is None:
            predict_list = ner_predict(model, text)
            if predict_list:
                if len(predict_list) > 1:
                    result = predict_list
                    if predict_list not in ORGS_DOUBT:
                        ORGS_DOUBT.append(predict_list)

                else:
                    result = format_orgname(predict_list[0])

        if result is not None:
            if isinstance(result, str):
                if result not in ORGS_LIST:
                    ORGS_LIST.append(result)
            df.at[i, prompt_data["output_column"]] = result

    df.to_csv(args.output, sep=prompt_data["delimiter"], index=False)

    ORGS_AMEND = []
    for values in ORGS_DOUBT:
        values = [format_orgname(v) for v in values]
        if not bool(set(values) & set(ORGS_LIST)):
            ORGS_AMEND.extend([val for val in values if val not in ORGS_AMEND])
            # if values not in ORGS_AMEND:
            #     ORGS_AMEND.append(values)

    if ORGS_AMEND:
        print("\n", end="")
        logger.info("Apure os dados:")
        ORGS_AMEND = prompt_amend_orgs(ORGS_AMEND)
        if ORGS_AMEND:
            ORGS_LIST.extend(
                [
                    val if val != "Nenhuma da Opções" else None
                    for val in ORGS_AMEND
                    if val not in ORGS_LIST
                ]
            )

    print("\n", end="")
    logger.info("Realizando pós-processamento...")
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        result = None
        str_output = str(row[prompt_data["output_column"]])
        if str_output.startswith("[") and str_output.endswith("]"):
            output_list = ast.literal_eval(str_output)
            for text in output_list:
                text = format_orgname(text)
                if next(
                    (
                        organization
                        for organization in ORGS_LIST
                        if organization in text
                    ),
                    None,
                ):
                    result = text
                    break

            df.at[i, prompt_data["output_column"]] = result

    df.to_csv(args.output, sep=prompt_data["delimiter"], index=False)

    print("\n", end="")
    logger.info(f"Total de empresas encontrada: {len(ORGS_LIST)}")


def ner_predict(model, text):
    doc = model(text)
    return [ent.text for ent in doc.ents]


def format_orgname(text: str):
    return re.sub(r"^[^\w]+|[^\w]+$", "", text).strip()


def get_prompt():
    response = dict()
    response["delimiter"] = args.sep if args.sep is not None else choose_delimiter()
    response["delimiter"] = {
        "TAB": "\t",
        "COMMA": ",",
        "SEMICOLON": ";",
        "PIPE": "|",
    }.get(response["delimiter"])

    response["table_start"] = (
        0
        if args.raw == "False"
        else detect_table_start(args.data, delimiter=response["delimiter"])
    )
    if response["table_start"] == -1:
        raise ValueError("Nenhuma tabela válida foi encontrada no arquivo CSV.")

    response["dataframe"] = pd.read_csv(
        args.data,
        sep=response["delimiter"],
        header=response["table_start"],
        low_memory=False,
        index_col=False,
    )

    response["column"] = (
        args.col if args.col is not None else choose_column(response["dataframe"])
    )
    if response["column"] is None:
        raise ValueError("Nenhuma coluna válida foi selecionada.")
    elif response["column"] not in response["dataframe"].columns:
        raise ValueError("A coluna selecionada não existe no DataFrame.")

    response["output_column"] = (
        args.out_col
        if args.out_col is not None
        else prompt_output_column(response["column"])
    )

    return response


def prompt_amend_orgs(orgs_list: list):
    return Check(
        "Quais destes são empresas? ",
        choices=orgs_list,
        check=" √",
        margin=2,
        check_color=colors.bright(colors.foreground["red"]),
        check_on_switch=colors.bright(colors.foreground["red"]),
        background_color=colors.background["black"],
        background_on_switch=colors.background["white"],
        word_color=colors.foreground["white"],
        word_on_switch=colors.foreground["black"],
    ).launch()


def choose_delimiter():
    DELIMITERS = {
        "Ponto e vírgula (;)": "SEMICOLON",
        "Vírgula (,)": "COMMA",
        "Tabulação (TAB)": "TAB",
        "Pipe (|)": "PIPE",
    }
    print("\n", end="")
    choices = [key for key in DELIMITERS.keys()]
    choices.append("Outro")
    delimiter = Bullet(
        prompt="Qual é o delimitador do seu CSV:\n",
        choices=choices,
        bullet="→ ",
    ).launch()

    if delimiter == "Outro":
        return str(Input("Digite o delimitador do seu CSV: ").launch()).strip()
    else:
        return DELIMITERS[delimiter]


def detect_table_start(csv_file: pathlib.Path, delimiter: str = "\t") -> int:
    """
    Detects the line where the table starts in a CSV file.

    Parameters:
    - csv_file (str): Path to the CSV file.
    - delimiter (str): Delimiter used in the CSV (default: comma).

    Returns:
    - (int) Index of the line where the table starts.
    """
    with open(csv_file, "r", encoding="UTF-8") as file:
        for index, line in enumerate(file):
            # Split the line by the delimiter and check how many columns exist
            columns = line.split(delimiter)
            if len(columns) > 1:
                if columns[0].strip() != "":
                    return index  # Return the line where the table starts
    return -1  # Return -1 if no valid line is found


def choose_column(df: pd.DataFrame):
    choices = list(df.columns.values)
    if len(choices) == 1:
        return choices[0]

    print("\n", end="")
    column = ScrollBar(
        prompt="Selecione a coluna:\n",
        choices=choices,
        height=5,
        align=0,
        margin=0,
        pointer="→ ",
    ).launch()
    return column


def prompt_output_column(input_column_name: str):
    default_name = f"{input_column_name}_OUTPUT"
    response = str(
        input(f"Qual será o nome de saída da coluna? (padrão: {default_name}): ")
    ).strip()
    if not response:
        response = default_name
    return response


if __name__ == "__main__":
    main()
