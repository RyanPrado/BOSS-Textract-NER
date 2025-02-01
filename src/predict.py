import spacy
from utils.logger import logger


def main():
    # Load English tokenizer, tagger, parser and NER
    spacy.prefer_gpu()
    # nlp = spacy.load("./models/sgs-ner-0.1.0a-5/model-best")
    nlp = spacy.load("pt_core_news_lg")

    labels = nlp.get_pipe("ner").labels

    print("Categorias reconhecidas pelo modelo:", labels)

    # Process whole documents
    text = "A empresa ACME INDUSTRIAL LTDA. assinou um contrato."
    doc = nlp(text)

    for entity in doc.ents:
        print(f"[{entity.label_}] - {entity.text}")


if __name__ == "__main__":
    main()
