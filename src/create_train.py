from sklearn.model_selection import train_test_split
import spacy
from spacy.tokens import DocBin
from utils.logger import logger
from io import TextIOWrapper
import pandas as pd

TRAIN_SIZE = 0.8


@logger.catch
def create_train(file_wrapper: TextIOWrapper):
    df = try_read_csv(file_wrapper)
    DATASET = convert_to_spacy_format(df)
    del df
    logger.info(f"Dataset Length: {len(DATASET)}")
    train_data, dev_data = train_test_split(
        DATASET, train_size=TRAIN_SIZE, random_state=42
    )
    logger.info(f"Treino: {len(train_data)} | Validação: {len(dev_data)}")
    save_spacy_file(train_data, "./data/corpus/train.spacy")
    save_spacy_file(dev_data, "./data/corpus/dev.spacy")


def save_spacy_file(data: list, file_path: str):
    nlp = spacy.blank("pt")
    db = DocBin()
    for text, annotations in data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)
            if span is not None:  # Só adiciona spans válidos
                ents.append(span)

        if ents:  # Só adiciona ao DocBin se houver entidades válidas
            doc.ents = ents
            db.add(doc)
    db.to_disk(file_path)


def try_read_csv(file_wrapper: TextIOWrapper):
    with file_wrapper as file:
        return pd.read_csv(
            file, delimiter=";", dtype={"FIRST_CHARACTER": int, "LAST_CHARACTER": int}
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
        description = row["GL_LINE_DESCRIPTION"].strip()
        entity_type = row["ENTITY_TYPE"].strip()
        entities.append(
            (
                description,
                [
                    (
                        row["FIRST_CHARACTER"],
                        row["LAST_CHARACTER"],
                        entity_type,
                    )
                ],
            )
        )
    return entities
