import argparse
from io import TextIOWrapper
import pandas as pd
from utils.logger import logger

MIN_SAMPLES = 5  # mínimo de exemplos por empresa

parser = argparse.ArgumentParser(
    prog="sciPy SGS NER Trainer",
    description="Utility for training a NER model",
)
parser.add_argument(
    "--data", type=argparse.FileType("r", encoding="UTF-8"), required=True
)
parser.add_argument("--output", type=argparse.FileType("w", encoding="UTF-8"))
args = parser.parse_args()


def main():
    df = try_read_csv(args.data)
    process_dataframe(
        df,
        args.output
        if isinstance(args.output, TextIOWrapper)
        else "./data/prepared/train.csv",
    )


def process_dataframe(df_input: pd.DataFrame, output_file: str):
    """
    Processa um DataFrame para gerar um novo com as colunas desejadas e salva em CSV.

    Parâmetros:
        df_input (pd.DataFrame): DataFrame de entrada contendo 'GL_LINE_DESCRIPTION' e 'NAME'.
        output_file (str): Nome do arquivo CSV de saída.

    Retorna:
        pd.DataFrame: DataFrame processado.
    """
    # Lista para armazenar os resultados
    results = []

    # Garanta que cada empresa tenha exemplos suficientes

    df_input["GL_LINE_DESCRIPTION"] = df_input["GL_LINE_DESCRIPTION"].str.upper()
    df_input["GL_LINE_DESCRIPTION"] = df_input["GL_LINE_DESCRIPTION"].str.strip()
    df_input["GL_LINE_DESCRIPTION"] = df_input["GL_LINE_DESCRIPTION"].replace(
        r"\s+", " ", regex=True
    )
    df_input["GL_LINE_DESCRIPTION"] = df_input["GL_LINE_DESCRIPTION"].replace(
        r"S\.A\.", "SA", regex=True
    )
    df_input["GL_LINE_DESCRIPTION"] = df_input["GL_LINE_DESCRIPTION"].replace(
        r"S/A", "SA", regex=True
    )
    df_input["GL_LINE_DESCRIPTION"] = df_input["GL_LINE_DESCRIPTION"].replace(
        r"EPP\.", "EPP", regex=True
    )
    df_input["GL_LINE_DESCRIPTION"] = df_input["GL_LINE_DESCRIPTION"].replace(
        r"ME\.", "ME", regex=True
    )

    df_input["GL_LINE_DESCRIPTION"] = df_input["GL_LINE_DESCRIPTION"].replace(
        r"LTDA.", "LTDA", regex=True
    )
    df_input["NAME"] = df_input["NAME"].str.upper()
    df_input["NAME"] = df_input["NAME"].str.strip()
    df_input["NAME"] = df_input["NAME"].replace(r"\s+", " ", regex=True)
    df_input["NAME"] = df_input["NAME"].replace(r"LTDA.", "LTDA", regex=True)
    df_input["NAME"] = df_input["NAME"].replace(r"S\.A\.", "SA", regex=True)
    df_input["NAME"] = df_input["NAME"].replace(r"S/A", "SA", regex=True)
    df_input["NAME"] = df_input["NAME"].replace(r"EPP\.", "EPP", regex=True)
    df_input["NAME"] = df_input["NAME"].replace(r"ME\.", "ME", regex=True)

    df_input = df_input.groupby("NAME").filter(lambda x: len(x) >= MIN_SAMPLES)

    for _, row in df_input.iterrows():
        text = row["GL_LINE_DESCRIPTION"]
        name = row["NAME"]
        # Encontrar a posição do nome dentro do texto
        start_idx = text.find(name) - 1
        if start_idx != -1:
            end_idx = (start_idx + len(name)) + 1  # Último caractere da entidade
            results.append(
                {
                    "GL_LINE_DESCRIPTION": text,
                    "FIRST_CHARACTER": start_idx,
                    "LAST_CHARACTER": end_idx,
                    "ENTITY_TYPE": "ORG",
                }
            )
    df_output = pd.DataFrame(results)
    logger.info(
        f"Amostras Validas({(df_output.shape[0] / df_input.shape[0]) * 100:.2f}%): {df_output.shape[0]} | Amostras Perdidas({((df_input.shape[0] - df_output.shape[0]) / df_input.shape[0]) * 100:.2f}%): {df_input.shape[0] - df_output.shape[0]}"
    )
    df_output.to_csv(output_file, index=False, encoding="utf-8", sep=";")


def try_read_csv(file_wrapper: TextIOWrapper):
    with file_wrapper as file:
        return pd.read_csv(file, sep=";", encoding="utf-8", index_col=False)


if __name__ == "__main__":
    main()
