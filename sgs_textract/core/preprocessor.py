from os import sep
import pathlib
import time
import pandas as pd
from torch import index_copy
from tqdm import tqdm
from utils.logger import logger


class DataPreprocessor:
    @staticmethod
    def _get_dataframe_from_csv(
        file_path: pathlib.Path, separator: str = ";", encoding: str = "UTF-8"
    ):
        with open(file_path, mode="r", encoding=encoding) as file:
            return pd.read_csv(file, sep=separator, encoding=encoding, index_col=False)

    @staticmethod
    def _format_column(column):
        column = column.str.upper()
        column = column.str.strip()

        return column

    @classmethod
    def create_train_dataframe(
        cls, df: pd.DataFrame, source_col: str, response_col: str, MIN_SAMPLES: int = 5
    ):
        df[source_col] = cls._format_column(df[source_col])
        df[response_col] = cls._format_column(df[response_col])
        df = df.groupby(response_col).filter(lambda x: len(x) >= MIN_SAMPLES)

        result_list = []

        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            source_val = row[source_col]
            response_val = row[response_col]
            # Encontrar a posição do nome dentro do texto
            start_idx = source_val.find(response_val) - 1
            if start_idx != -1:
                end_idx = (
                    start_idx + len(response_val)
                ) + 1  # Último caractere da entidade
                result_list.append(
                    {
                        "SOURCE": source_val,
                        "FIRST_CHAR": start_idx,
                        "LAST_CHAR": end_idx,
                        "ENTITY_TYPE": "MISC",
                    }
                )
        return pd.DataFrame(result_list)

    @classmethod
    def load(cls, file_path: pathlib.Path, separator: str, encoding: str):
        if not file_path.is_file():
            raise TypeError("O arquivo não pode ser um diretório")

        df = cls._get_dataframe_from_csv(file_path, separator, encoding)

        logger.info(f"Pré-processando os dados! {file_path}")
        return df
