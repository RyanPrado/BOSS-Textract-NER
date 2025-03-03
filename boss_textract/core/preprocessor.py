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
                on_bad_lines='warn'
            )

    @staticmethod
    def format_column(column):
        column = column.str.upper()
        column = column.replace(r"<BR>", " ", regex=True)
        column = column.replace(r"\s+", " ", regex=True)
        # Ending to [CPF,CPNJ or Date] -> Empty
        column = column.replace(r"\d{11,}|\d{1,2}[\/\\]\d{2,4}$", "", regex=True)
        # [ /NAME, NAME/] -> NAME
        column = column.replace(
            r"(?:\s|^)[\/\\](?=[\w\s\.\,])|(?<=[\w\s\.\,])[\/\\](?=\s|$)",
            " ",
            regex=True,
        )
        # [ENTER-NAME] -> ENTER NAME
        column = column.replace(
            r"(?<=[\w\s\.\,])\-(?=[\w\s\.\,])",
            " - ",
            regex=True,
        )
        # [GOOGLE,LLC] -> GOOGLE LLC
        column = column.replace(
            r"(?<=[\w\s\.\,])(?<!\s)[\\\/,\;]+(?=(?=\s)|(?!\s))(?<![\w\s])",
            " ",
            regex=True,
        )

        # [ - ME, - EM, - EPP] -> Empty
        column = column.replace(r"\s?\-\s*(ME|EM|EPP)(?=$|\s+)", "", regex=True)
        # [ S.] -> Empty
        column = column.replace(r"\s+S\.(\s+|$)", "", regex=True)
        # [B.V.] -> B.V
        column = column.replace(r"B\.V\.?(\s+|$)", "B.V ", regex=True)
        # [U.A.] -> U.A
        column = column.replace(r"U\.A\.?(\s+|$)", "U.A ", regex=True)
        # [A.S.] -> A.S
        column = column.replace(r"A\.S\.?(\s+|$)", "A.S ", regex=True)
        # [N.V.] -> N.V
        column = column.replace(r"N\.V\.?(\s+|$)", "N.V ", regex=True)
        # [Z.O.O.] -> Z.O.O
        column = column.replace(
            r"Z[\.\s]+O[\.\s]+O\.?[\/\\]?(\s+|$)", "Z.O.O ", regex=True
        )
        # [S.P.A.] -> S.P.A
        column = column.replace(
            r"S[\.\s]+P[\.\s]+A\.?[\/\\]?(\s+|$)", "S.P.A ", regex=True
        )
        # [S.R.O.] -> S.R.O
        column = column.replace(
            r"S[\.\s]+R[\.\s]+O\.?[\/\\]?(\s+|$)", "S.R.O ", regex=True
        )
        # [S.A.C.] -> S.A.C
        column = column.replace(
            r"S[\.\s]+A[\.\s]+C\.?[\/\\]?(\s+|$)", "S.A.C ", regex=True
        )
        # [S.A.S.] -> S.A.S
        column = column.replace(
            r"S[\.\s]+A[\.\s]+S\.?[\/\\]?(\s+|$)", "S.A.S ", regex=True
        )
        # [S.A.U.] -> S.A.U
        column = column.replace(
            r"S[\.\s]+A[\.\s]+U\.?[\/\\]?(\s+|$)", "S.A.U ", regex=True
        )
        # [S.P.A.] -> S.P.A
        column = column.replace(
            r"S[\.\s]+R[\.\s]+L\.?[\/\\]?(\s+|$)", "S.R.L ", regex=True
        )
        # [NF.1234-, NF.12345,NF.] -> Empty
        column = column.replace(r"(?:NF\.)(?:(?:\d+\-?)|(?=[\w\d]))", "", regex=True)
        # [RPS: 1234, RPS 1234] -> Empty
        column = column.replace(r"(?:RPS:?\s)(?:\d+\-?)", "", regex=True)
        # [LTDA- ME,LTDA- EM,LTDA - EPP, LTDAME] -> LTDA
        column = column.replace(
            r"LTDA\.?\s?[\.\-\|]?(?=\w*)\s*(ME|EM|EPP|\/\d{2,4})?\s*",
            "LTDA ",
            regex=True,
        )
        # [S.A., S\A, S/A, SA] -> S.A
        column = column.replace(
            r"(?<=\s)S\.?/?\\?\s?A\.?(?:C\.)?(?=\s+|$)", "S.A", regex=True
        )
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
