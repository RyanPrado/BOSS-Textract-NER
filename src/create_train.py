import pathlib
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import spacy
from spacy.tokens import DocBin
from utils.logger import logger
from io import TextIOWrapper
import pandas as pd

TRAIN_SIZE = 0.8


@logger.catch
def create_train(file_wrapper: TextIOWrapper, model_path: pathlib.Path):
    df = try_read_csv(file_wrapper)
    df = shuffle(df)
    DATASET = convert_to_spacy_format(df)
    del df
    logger.info(f"Dataset Length: {len(DATASET)}")
    train_data, dev_data = train_test_split(
        DATASET, train_size=TRAIN_SIZE, random_state=42
    )
    logger.info(f"Treino: {len(train_data)} | Validação: {len(dev_data)}")
    logger.debug(f"Treino último item:\n{train_data[len(train_data) - 1]}")
    logger.debug(f"Validação último item:\n{dev_data[len(dev_data) - 1]}")
    if model_path is not None:
        logger.debug(f"Realizando transfer-learning para o modelo em {str(model_path)}")

    save_spacy_file(train_data, "./data/corpus/train.spacy", model_path=model_path)
    save_spacy_file(dev_data, "./data/corpus/dev.spacy", model_path=model_path)


def save_spacy_file(TRAIN_DATA: list, file_path: str, model_path: str = None):
    nlp = None
    if model_path is not None:
        nlp = spacy.load(model_path)
    if nlp is None:
        nlp = spacy.blank("pt")

    db = DocBin()
    invalid_spans = 0
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        ents = []
        logger.trace("=" * 50)
        logger.trace(f"Doc: {text}")
        logger.trace(f"Annotations: {annotations}")
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            logger.trace(f"Span: {span} | {str(doc)[start:end]}")
            if span is not None:
                ents.append(span)
            else:
                invalid_spans += 1
        logger.trace("=" * 50)
        # time.sleep(2.5)

        if ents:
            doc.ents = ents
            db.add(doc)

    logger.info(f"Foram encontrados {invalid_spans} spans inválidos em '{file_path}'")
    db.to_disk(file_path)


def try_read_csv(file_wrapper: TextIOWrapper):
    with file_wrapper as file:
        return pd.read_csv(
            file,
            sep=";",
            index_col=False,
            header=0,
            low_memory=False,
            encoding="utf-8",
            dtype={
                "GL_LINE_DESCRIPTION": str,
                "FIRST_CHARACTER": int,
                "LAST_CHARACTER": int,
                "ENTITY_TYPE": str,
            },
        )


def convert_to_spacy_format(data: pd.DataFrame):
    """
    Converts a pandas DataFrame to a format compatible with spaCy for named entity recognition (NER) training.
    Args:
        data (pd.DataFrame): A DataFrame containing the following columns:
            - 'GL_LINE_DESCRIPTION': The text containing the entities.
            - 'FIRST_CHARACTER': The starting character index of the entity in the text.
            - 'LAST_CHARACTER': The ending character index of the entity in the text.
            - 'ENTITY_TYPE': The type of the entity (e.g., 'PERSON', 'ORG', etc.).
    Returns:
        List[Tuple[str, List[Tuple[int, int, str]]]]: A list of tuples where each tuple contains:
            - The text (str) from 'GL_LINE_DESCRIPTION'.
            - A list of tuples, each containing:
                - The starting character index (int) of the entity.
                - The ending character index (int) of the entity.
                - The entity type (str).
            - Response (str) extract from 'GL_LINE_DESCRIPTION' based in first/last character position.

    """

    entities = []
    for _, row in data.iterrows():
        description = str(row["GL_LINE_DESCRIPTION"]).strip()
        entity_type = str(row["ENTITY_TYPE"]).strip()
        entities.append(
            (
                description,
                [
                    (
                        int(row["FIRST_CHARACTER"]),
                        int(row["LAST_CHARACTER"]),
                        entity_type,
                    )
                ],
            )
        )
        # logger.debug(f"{entities[len(entities) - 1]}")
    return entities
