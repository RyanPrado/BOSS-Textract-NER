from pathlib import Path
import sys

from utils._version import __version__
import pandas as pd
from core.preprocessor import DataPreprocessor
from core.trainer import ModelTrainer
from utils.logger import logger
import glob
import re
from commands.base_command import BaseCommand
from spacy.util import load_config

from bullet import Bullet, Input, ScrollBar, SlidePrompt, Check, colors

SEPARATORS = {
    "TAB": "\t",
    "\\t": "\t",
    "COMMA": ",",
    ",": ",",
    "SEMICOLON": ";",
    ";": ";",
    "PIPE": "|",
    "|": "|",
}


class MissingColumnError(Exception): ...


class EmptyResponseError(Exception): ...


class TrainCommand(BaseCommand):
    @classmethod
    def _get_columns(cls, df: pd.DataFrame, args):
        df_columns = list(df.columns.values)

        src_column = args.src_col
        res_columns = []
        if args.res_col is not None and isinstance(args.res_col, str):
            splits_columns = args.res_col.split(";")
            res_columns = [
                (lambda x: {"column": x[0], "type": x[1]})(x.split(":"))
                for x in splits_columns
            ]

        listed_columns = [y["column"] for y in res_columns]
        src_column = (
            src_column
            if src_column is not None
            else cls._choose_column(
                "Selecione a coluna de origem",
                [x for x in df_columns if (x not in listed_columns)],
            )
        )

        if len(res_columns) == 0:
            while True:
                listed_columns = [y["column"] for y in res_columns]
                opts_columns = [
                    x
                    for x in df_columns
                    if (x != src_column) and (x not in listed_columns)
                ]
                selected_column = None
                if opts_columns == 0:
                    break
                opts_columns.append("(Finalizar)")
                selected_column = cls._choose_column(
                    "Selecione uma coluna", opts_columns
                )
                if selected_column == "(Finalizar)":
                    break
                res_columns.append(
                    {
                        "column": selected_column,
                        "type": cls._input_text(
                            f"Digite o LABEL da coluna:\nSEUS LABELS ATUAIS [{','.join([x['type'] for x in res_columns])}]"
                        ),
                    }
                )

        if src_column not in df.columns:
            raise MissingColumnError(src_column)

        if len(res_columns) == 0:
            raise EmptyResponseError(res_columns)

        for x_column in res_columns:
            if x_column["column"] not in df.columns:
                raise MissingColumnError(x_column)
            elif x_column["column"] == src_column:
                raise TypeError(
                    "Coluna de origem não pode ser também uma coluna de resposta"
                )

        return src_column, res_columns

    @staticmethod
    def _choose_column(question: str, columns: list):
        if columns is not None and len(columns) == 0:
            raise ValueError("Não a colunas disponíveis para seleção")
        elif len(columns) == 1:
            return columns[0]

        print("\n", end="")
        return ScrollBar(
            prompt=f"{question}:\n",
            choices=columns,
            height=5,
            align=0,
            margin=0,
            pointer="→ ",
        ).launch()

    @staticmethod
    def _input_text(question: str):
        print("\n", end="")
        print(f"{question}\n")
        return str(input()).upper()

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--data", type=Path, required=True)
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--model", type=Path)
        parser.add_argument("--src_col", type=str)
        parser.add_argument("--res_col", type=str)
        parser.add_argument("--regex", type=Path)
        parser.add_argument("--encoding", type=str, default="UTF-8")
        parser.add_argument("--sep", type=str, default="SEMICOLON")
        parser.add_argument("--min_samples", type=int, default=5)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--dropout", type=float)
        parser.add_argument(
            "--output",
            type=Path,
            default=f"./models/sgs-ner-{__version__}-{get_next_model_number(__version__)}",
        )

    @classmethod
    def execute(cls, args):
        try:
            file_path = args.data
            separator = SEPARATORS.get(args.sep)
            encoding = args.encoding
            min_samples = int(args.min_samples)
            separator = separator if separator is not None else args.sep

            if not file_path.is_file():
                raise TypeError(
                    "O arquivo selecionado é invalido para realizar o treinamento"
                )

            # Loading and formatting dataframe
            df_input = DataPreprocessor.load(file_path, separator, encoding)
            src_column, res_columns = cls._get_columns(df_input, args)

            logger.info(f"Coluna de origem: [{src_column}]")
            logger.info(f"Coluna c/ resposta: [{res_columns}]")

            logger.info("Iniciando criação do dataset de treino...")
            logger.info(f"Amostras listadas: {df_input.shape[0]}")
            dataset = DataPreprocessor.create_train_dataframe(
                df_input, src_column, res_columns, MIN_SAMPLES=min_samples
            )
            logger.info("Iniciando criação do dataset de treino...")
            logger.info(f"Amostras confirmadas: {len(dataset)}")

            config = load_config(args.config)

            overrides = {}
            overrides["training.dropout"] = (
                args.epochs
                if args.epochs is not None
                else config["training"]["max_epochs"]
            )
            overrides["training.dropout"] = (
                args.dropout
                if args.dropout is not None
                else config["training"]["dropout"]
            )

            if args.model:
                overrides["components.ner.source"] = str(args.model)
                overrides["components.transformer.source"] = str(args.model)
            else:
                overrides["components.ner.factory"] = "ner"
                overrides["components.transformer.factory"] = "transformer"

            trainer = ModelTrainer(args.config, args.output, overrides=overrides)
            trainer.train(dataset)

        except EmptyResponseError as e:
            logger.error("Nenhuma coluna de resposta foi selecionada")
        except MissingColumnError as e:
            logger.error(
                f"A Coluna {e.args[0]} não está presente no arquivo {file_path}"
            )

        except Exception as e:
            exception = sys.exc_info()
            logger.opt(exception=exception).error(e)


def get_next_model_number(version: str) -> str:
    pattern = re.compile(rf"sgs-ner-{version}-(\d+)")
    models = glob.glob("./models/*")
    numbers = [
        int(pattern.search(model).group(1)) for model in models if pattern.search(model)
    ]
    return max(numbers) + 1 if numbers else 1
