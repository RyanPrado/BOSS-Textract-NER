import pathlib
import pandas as pd
from tqdm import tqdm
from utils.logger import logger


class DataPreprocessor:
    @staticmethod
    def _get_dataframe_from_csv(
        file_path: pathlib.Path,
        separator: str = ";",
        encoding: str = "UTF-8",
        *,
        start_header: int,
    ):
        with open(file_path, mode="r", encoding=encoding) as file:
            return pd.read_csv(
                file,
                sep=separator,
                encoding=encoding,
                index_col=False,
                header=start_header,
            )

    @staticmethod
    def format_column(column):
        column = column.str.upper()
        column = column.replace(r"\s+", " ", regex=True)
        # [LTDA- ME,LTDA- EM,LTDA - EPP, LTDAME] -> LTDA
        column = column.replace(
            r"LTDA\.?\s?[\.\-\|]?\w*\s*(ME|EM|EPP|\/\d{2,4})?", "LTDA", regex=True
        )
        # [ - ME, - EM, - EPP] -> Empty
        column = column.replace(r"\s?\-\s*(ME|EM|EPP)(?=$|\s+)", "", regex=True)
        # [S.A., S\A, S/A, SA] -> S.A
        column = column.replace(
            r"(?<=\s)S\.?/?\\?\s?A\.?(?:C\.)?(?=$|\s+)", "S.A", regex=True
        )
        # Ending to [CPF,CPNJ or Date] -> Empty
        column = column.replace(r"\d{11,}$|\d{1,2}\/\d{2,4}$", "", regex=True)
        column = column.replace(r"\s+", " ", regex=True)
        column = column.str.strip()

        return column

    @classmethod
    def create_train_dataframe(
        cls, df: pd.DataFrame, source_col: str, response_cols: str, MIN_SAMPLES: int = 5
    ):
        df[source_col] = cls.format_column(df[source_col])
        for response_col in response_cols:
            df[response_col["column"]] = cls.format_column(df[response_col["column"]])
            df = df.groupby(response_col["column"]).filter(
                lambda x: len(x) >= MIN_SAMPLES
            )

        training_data = []

        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            source_val = row[source_col]
            train_element = [source_val, {"entities": []}]
            for response_col in response_cols:
                response_val = row[response_col["column"]]
                start_idx = source_val.find(response_val)
                if start_idx != -1:
                    end_idx = start_idx + len(response_val)
                    train_element[1]["entities"].append(
                        (start_idx, end_idx, response_col["type"])
                    )

            if len(train_element[1]["entities"]) > 0:
                training_data.append(train_element)

        return training_data

    @classmethod
    def load(
        cls,
        file_path: pathlib.Path,
        separator: str,
        encoding: str,
        *,
        start_header: int = 0,
    ):
        if not file_path.is_file():
            raise TypeError("O arquivo não pode ser um diretório")

        df = cls._get_dataframe_from_csv(
            file_path, separator, encoding, start_header=start_header
        )

        logger.info(f"Pré-processando os dados! {file_path}")
        return df
