from pathlib import Path
import tempfile
import spacy
from sklearn.model_selection import train_test_split
from utils.logger import logger
from typing import Union, Dict, Any
from spacy.cli.train import train


class ModelTrainer:
    def __init__(
        self,
        config_path: Union[Path, str],
        output_path: Union[Path, str],
        *,
        overrides: Dict[str, Any] = spacy.util.SimpleFrozenDict(),
    ):
        self.config_path = spacy.util.ensure_path(config_path)
        self.output_path = spacy.util.ensure_path(output_path)
        self.overrides = overrides

    def train(
        self,
        dataset: list[str, Dict[str, list[int, int, str]]],
        *,
        train_size: float = 0.8,
    ):
        train_data, dev_data = train_test_split(
            dataset, train_size=train_size, random_state=42
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = Path(temp_dir) / "train.spacy"
            dev_path = Path(temp_dir) / "dev.spacy"
            self._prepare_training_data(train_data, train_path)
            self._prepare_training_data(dev_data, dev_path)

            self.overrides["paths.train"] = str(train_path)
            self.overrides["paths.dev"] = str(dev_path)

            train(
                self.config_path,
                output_path=self.output_path,
                use_gpu=0,
                overrides=self.overrides,
            )

    def _prepare_training_data(self, dataset: list, output_path: Path):
        # Salvar dados em formato spaCy
        doc_bin = spacy.tokens.DocBin()
        nlp = spacy.blank("pt")
        invalid_spans = 0
        for text, annotations in dataset:
            doc = nlp.make_doc(text)
            ents = []

            for start, end, label in annotations["entities"]:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                logger.trace(f"Span: {span} | {str(doc)[start:end]}")
                if span is not None:
                    ents.append(span)
                else:
                    invalid_spans += 1

            doc.ents = ents
            doc_bin.add(doc)
        logger.info(f"Foram encontrados {invalid_spans} spans inválidos.")
        doc_bin.to_disk(output_path)
