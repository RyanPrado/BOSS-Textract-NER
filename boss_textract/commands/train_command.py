from email.policy import default
import math
from pathlib import Path
import sys
from typing import Union
import os
from utils._version import __version__
import pandas as pd
from core.preprocessor import DataPreprocessor
from core.trainer import ModelTrainer
from utils.logger import logger
from utils import SEPARATORS
import glob
import re
from commands.base_command import BaseCommand
from spacy.util import load_config

if os.name != "nt":
    from bullet import ScrollBar


class MissingColumnError(Exception): ...


class EmptyResponseError(Exception): ...


class TrainCommand(BaseCommand):
    @classmethod
    def _has_columns(cls, df: pd.DataFrame, columns: Union[str, list[str]]):
        if isinstance(columns, str):
            columns = [columns]

        if len(columns) == 0:
            return True

        return all([x if x in df.columns else False for x in columns])

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
                opts_columns.append("[ Finalizar ]")
                selected_column = cls._choose_column(
                    "Selecione uma coluna", opts_columns
                )
                if selected_column == "[ Finalizar ]":
                    break
                res_columns.append(
                    {
                        "column": selected_column,
                        "type": cls._input_text(
                            f"Digite o LABEL da coluna:\nSEUS LABELS ATUAIS [{','.join([x['type'] for x in res_columns])}]\n"
                        ),
                    }
                )

        return src_column, res_columns

    @staticmethod
    def _choose_column(question: str, columns: list):
        if columns is not None and len(columns) == 0:
            raise ValueError("Não a colunas disponíveis para seleção")
        elif len(columns) == 1:
            return columns[0]
        if os.name == "nt":
            return None

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
        print(f"{question}", end="")
        return str(input()).upper()

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--data", type=Path, required=True)
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--eval", type=Path)
        parser.add_argument("--model", type=Path)
        parser.add_argument("--src_col", type=str)
        parser.add_argument("--res_col", type=str)
        parser.add_argument("--encoding", type=str, default="UTF-8")
        parser.add_argument("--sep", type=str, default="SEMICOLON")
        parser.add_argument("--min_samples", type=int, default=5)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--gpu_id", type=int, default=-1)
        parser.add_argument("--train_size", type=float, default=0.8)
        parser.add_argument("--dropout", type=float)
        parser.add_argument("--eval_frequency", type=int)
        parser.add_argument(
            "--output",
            type=Path,
            default=f"./models/boss-ner-{__version__}-{get_next_model_number(__version__)}",
        )

    @classmethod
    def execute(cls, args):
        try:
            file_path = args.data
            eval_path = args.eval
            train_size = args.train_size
            separator = SEPARATORS.get(args.sep)
            encoding = args.encoding
            min_samples = int(args.min_samples)
            separator = separator if separator is not None else args.sep

            if not file_path.is_file():
                raise TypeError(
                    "O arquivo selecionado é invalido para realizar o treinamento"
                )
            if eval_path is not None and not eval_path.is_file():
                raise TypeError(
                    "O arquivo selecionado é invalido para realizar o treinamento"
                )

            if separator is None:
                raise ValueError(f"O separador {separator} é invalido")

            # Loading and formatting dataframe
            df_input = DataPreprocessor.load(file_path, separator, encoding)
            df_eval = (
                DataPreprocessor.load(eval_path, separator, encoding)
                if eval_path is not None
                else None
            )

            src_column, res_columns = cls._get_columns(df_input, args)
            if len(res_columns) == 0:
                raise EmptyResponseError(res_columns)

            # Validation columns in CSV files
            for v in (
                {"df": df_input, "path": file_path},
                {"df": df_eval, "path": eval_path},
            ):
                if v["df"] is not None:
                    if cls._has_columns(v["df"], src_column) is False:
                        raise MissingColumnError(src_column, v["path"])
                    if (
                        cls._has_columns(v["df"], [x["column"] for x in res_columns])
                        is False
                    ):
                        raise MissingColumnError(
                            [x["column"] for x in res_columns], v["path"]
                        )

            logger.info(f"Coluna de origem: [{src_column}]")
            logger.info(f"Coluna c/ resposta: [{res_columns}]")

            logger.info("Iniciando criação do dataset de treino...")
            logger.info(
                f"Amostras listadas: Treino [{df_input.shape[0] if df_eval is not None else math.ceil(df_input.shape[0] * train_size)}] | Validação [{df_eval.shape[0] if df_eval is not None else math.ceil(df_input.shape[0] * (1 - train_size))}]"
            )

            if df_eval is not None:
                df_input = df_input[~df_input[src_column].isin(df_eval[src_column])]

            dataset = DataPreprocessor.create_train_dataframe(
                df_input, src_column, res_columns, MIN_SAMPLES=min_samples
            )

            eval_dataset = (
                DataPreprocessor.create_train_dataframe(
                    df_eval, src_column, res_columns, MIN_SAMPLES=1
                )
                if df_eval is not None
                else None
            )
            del df_eval, df_input

            logger.info("Iniciando criação do dataset de treino...")
            logger.info(
                f"Amostras de Apuradas: Treino [{len(dataset) if eval_dataset is not None else math.ceil(len(dataset) * train_size)}] | Validação [{len(eval_dataset) if eval_dataset is not None else math.ceil(len(dataset) * (1 - train_size))}]"
            )

            config = load_config(args.config)

            overrides = {}
            overrides["training.max_epochs"] = (
                args.epochs
                if args.epochs is not None
                else config["training"]["max_epochs"]
            )
            overrides["training.eval_frequency"] = (
                args.eval_frequency
                if args.eval_frequency is not None
                else config["training"]["eval_frequency"]
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
            trainer.train(
                dataset,
                train_size=train_size,
                gpu_id=args.gpu_id,
                dev_data=eval_dataset,
            )

        except EmptyResponseError as e:
            logger.error("Nenhuma coluna de resposta foi selecionada")
        except MissingColumnError as e:
            logger.error(
                f"A Coluna {e.args[0]} não está presente no arquivo {e.args[1]}"
            )

        except Exception as e:
            exception = sys.exc_info()
            logger.opt(exception=exception).error(e)


def get_next_model_number(version: str) -> str:
    pattern = re.compile(rf"boss-ner-{version}-(\d+)")
    models = glob.glob("./models/*")
    numbers = [
        int(pattern.search(model).group(1)) for model in models if pattern.search(model)
    ]
    return max(numbers) + 1 if numbers else 1
