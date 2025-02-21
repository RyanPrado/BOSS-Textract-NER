from pathlib import Path
import sys
from typing import Union

from spacy.util import ensure_path
from utils.logger import logger, log_path
from commands.base_command import BaseCommand
from utils import SEPARATORS
from core.preprocessor import DataPreprocessor
from core.predicter import ModelPredicter
from bullet import Bullet, Input, ScrollBar
from spacy import displacy
import time


class PredictCommand(BaseCommand):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--data", type=Path, required=True)
        parser.add_argument("--model", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--src_col", type=str)
        parser.add_argument("--out_col", type=str)
        parser.add_argument("--start_header", type=int)
        parser.add_argument("--sep", type=str)
        parser.add_argument("--max_variation", type=int, default=0)
        parser.add_argument("--encoding", type=str, default="UTF-8")
        parser.add_argument("--gpu_id", type=int, default=-1)
        parser.add_argument("--log", type=Path, default=Path(log_path))
        parser.add_argument("--no-log", dest="log", action="store_false")

    @staticmethod
    def _get_start_header_index_by_column(
        file_path: Path, sep: str, column_name: str, encoding: str = "UTF-8"
    ):
        with open(file_path, "r", encoding=encoding) as file:
            for index, line in enumerate(file):
                columns = line.split(sep)
                if (column_name is not None and column_name in columns) or len(
                    columns
                ) > 1:
                    if columns[0].strip() != "":
                        return index
            return 0

    @staticmethod
    def _input_text(question: str):
        print("\n", end="")
        print(f"{question}", end="")
        return str(input()).upper()

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
    def _choose_separator():
        SEPARATORS_TITLES = {
            "Ponto e vírgula (;)": "SEMICOLON",
            "Vírgula (,)": "COMMA",
            "Tabulação (TAB)": "TAB",
            "Pipe (|)": "PIPE",
        }
        print("\n", end="")
        choices = [key for key in SEPARATORS_TITLES.keys()]
        choices.append("[ Outro ]")
        separator = Bullet(
            prompt="Qual é o delimitador do seu CSV:\n",
            choices=choices,
            bullet="→ ",
        ).launch()

        if separator == "[ Outro ]":
            return str(Input("Digite o delimitador do seu CSV: ").launch()).strip()
        else:
            return SEPARATORS.get(SEPARATORS_TITLES[separator])

    @classmethod
    def execute(cls, args):
        try:
            file_path: Path = args.data
            separator: str = SEPARATORS.get(args.sep)
            model_path: Path = args.model
            encoding: str = args.encoding
            separator = separator if separator is not None else args.sep
            separator = separator if separator is not None else cls._choose_separator()
            source_column = args.src_col if args.src_col is not None else None
            if not file_path.is_file():
                raise TypeError("O arquivo selecionado é invalido.")
            if separator is None:
                raise ValueError(f"O separador {separator} é invalido")

            start_header = (
                args.start_header
                if args.start_header is not None
                else cls._get_start_header_index_by_column(
                    file_path, separator, column_name=source_column, encoding=encoding
                )
            )
            df = DataPreprocessor.load(
                file_path, separator, encoding, start_header=start_header
            )
            source_column = (
                source_column
                if source_column is not None
                else cls._choose_column(
                    "Selecione a coluna de origem", list(df.columns.values)
                )
            )
            if source_column is None:
                raise ValueError("Nenhuma coluna válida foi selecionada.")
            elif source_column not in df.columns:
                raise ValueError("A coluna selecionada não existe no DataFrame.")

            output_column = (
                args.out_col
                if args.out_col is not None
                else cls._input_text(
                    f"Digite o nome da coluna de saída (Padrão: {source_column.upper()}_OUTPUT): "
                )
            )
            if output_column is None or output_column == "":
                output_column = f"{source_column.upper()}_OUTPUT"

            log_path: Union[Path, bool] = args.log
            if isinstance(log_path, Path) and Path(log_path).is_dir():
                log_path = log_path / Path(
                    f"log_{time.strftime('%Y-%m-%d_%H-%M-%S')}.html"
                )
            if log_path:
                logger.info(f"Logging in: {log_path.absolute()}")

            predicter = ModelPredicter(model_path, gpu_id=args.gpu_id)
            df, docs = predicter.predict(
                df,
                source_column,
                output_column,
                max_variation=args.max_variation,
                log=isinstance(log_path, Path),
            )
            output_path: Path = ensure_path(args.output)

            if output_path.is_file():
                output_path = output_path.with_suffix(".csv")
            elif output_path.is_dir():
                output_path = output_path / f"{file_path.stem}.csv"

            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output directory: {output_path.parent}")

            logger.success(f"Saving output file: {output_path}")
            df.to_csv(output_path, sep=separator, index=False)
            del df

            if docs and isinstance(log_path, Path):
                svg = displacy.render(docs, style="ent", page=True, minify=True)

                if not log_path.parent.exists():
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.open("w", encoding="utf-8").write(svg)

        except Exception as e:
            exception = sys.exc_info()
            logger.opt(exception=exception).error(e)
