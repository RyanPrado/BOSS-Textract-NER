import gc
from pathlib import Path
import tempfile
import spacy
from sklearn.model_selection import train_test_split
from utils.logger import logger
from typing import Union, Dict, Any
from spacy.cli.train import train
from tqdm import tqdm


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
        train_data: list[str, Dict[str, list[int, int, str]]],
        *,
        train_size: float = 0.8,
        dev_data: list[str, Dict[str, list[int, int, str]]] = None,
        gpu_id: int = -1,
    ):
        if dev_data is None:
            train_data, dev_data = train_test_split(
                train_data, train_size=train_size, random_state=42
            )

        gc.collect()
        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = Path(temp_dir) / "train.spacy"
            dev_path = Path(temp_dir) / "dev.spacy"
            logger.info("Preparing Training data")
            invalid_spans = 0
            invalid_spans += self._prepare_training_data(
                train_data, train_path, "TRAIN DATASET"
            )
            invalid_spans += self._prepare_training_data(
                dev_data, dev_path, "EVAL DATASET"
            )
            logger.info(f"Foram encontrados {invalid_spans} spans inválidos.")

            self.overrides["paths.train"] = str(train_path)
            self.overrides["paths.dev"] = str(dev_path)

            del train_data, dev_data
            train(
                self.config_path,
                output_path=self.output_path,
                use_gpu=gpu_id,
                overrides=self.overrides,
            )

    def _prepare_training_data(self, dataset: list, output_path: Path, debug_name: str):
        # Salvar dados em formato spaCy
        doc_bin = spacy.tokens.DocBin()
        nlp = spacy.blank("pt")
        invalid_spans = 0
        for text, annotations in tqdm(dataset):
            doc = nlp.make_doc(text)
            ents = []

            for start, end, label in annotations["entities"]:
                span = doc.char_span(start, end, label=label, alignment_mode="strict")
                if span is not None:
                    logger.trace(
                        f"Span: {str(span)} | Text: {str(doc)[start:end]} | Source: {str(doc)}"
                    )
                    ents.append(span)
                else:
                    logger.debug(
                        f"{f'[{debug_name}] ' if debug_name is not None else ''}Source don't has char_span [{str(doc)}] <- [{str(doc)[start:end]}]"
                    )
                    invalid_spans += 1

            doc.ents = ents
            doc_bin.add(doc)
        doc_bin.to_disk(output_path)
        return invalid_spans
