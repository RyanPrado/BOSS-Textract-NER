import argparse
from io import TextIOWrapper
import pandas as pd


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
    print(type(args.output))
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

    for _, row in df_input.iterrows():
        text = row["GL_LINE_DESCRIPTION"]
        name = row["NAME"]

        results = []
        # Encontrar a posição do nome dentro do texto
        start_idx = text.find(name)
        if start_idx != -1:
            end_idx = start_idx + len(name) - 1  # Último caractere da entidade
            results.append(
                {
                    "GL_LINE_DESCRIPTION": text,
                    "FIRST_CHARACTER": start_idx,
                    "LAST_CHARACTER": end_idx,
                    "ENTITY_TYPE": "ORG",
                }
            )
        df_output = pd.DataFrame(results)
        df_output.to_csv(output_file, index=False, encoding="utf-8", sep=";")


def try_read_csv(file_wrapper: TextIOWrapper):
    with file_wrapper as file:
        return pd.read_csv(file, delimiter=";")


if __name__ == "__main__":
    main()
