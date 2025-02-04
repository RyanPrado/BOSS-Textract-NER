import torch
from spacy.cli.train import train
from _version import __version__
from create_train import create_train
import argparse
import re
import glob


from spacy.util import load_config

config = load_config("./config.cfg")

if torch.cuda.is_available():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    prog="sciPy SGS NER Trainer",
    description="Utility for training a NER model",
)
parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
parser.add_argument(
    "--data", type=argparse.FileType("r", encoding="UTF-8"), required=True
)
parser.add_argument("--epochs", type=int)
args = parser.parse_args()

"""
Objetivo:
    Crie uma função para gerar o próximo número do modelo

Informações:
    - O número do modelo é gerado a partir do diretório `models`
    - O padrão de texto é sgs-textract-{VERSION}.{NUMERO}
    - {VERSION} é a versão do modelo
    - {NUMERO} é a geração do modelo
    - A versão do modelo é a variável __version__
"""


def main():
    create_train(args.data)
    train(
        "./config.cfg",
        use_gpu=int("0" if device.type == "cuda" else "-1"),
        output_path=f"./models/sgs-ner-{__version__}-{get_next_model_number(__version__)}",
        overrides={
            "training.max_epochs": args.epochs
            if args.epochs
            else config["training"]["max_epochs"],
        },
    )


def get_next_model_number(version: str) -> str:
    pattern = re.compile(rf"sgs-ner-{version}-(\d+)")
    models = glob.glob("./models/*")
    numbers = [
        int(pattern.search(model).group(1)) for model in models if pattern.search(model)
    ]
    return max(numbers) + 1 if numbers else 1


if __name__ == "__main__":
    main()
