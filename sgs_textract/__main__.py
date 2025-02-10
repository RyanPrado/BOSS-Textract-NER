import argparse
from utils.logger import logger
from commands.train_command import TrainCommand
from commands.predict_command import PredictCommand

COMMANDS = {
    "train": TrainCommand,
    "predict": PredictCommand,
}


def main():
    parser = argparse.ArgumentParser(prog="SGS Textract NER")
    subparsers = parser.add_subparsers(dest="command")

    for command_name, command_cls in COMMANDS.items():
        command_instance = command_cls()
        subparser = subparsers.add_parser(command_name)
        command_instance.add_arguments(subparser)

    args = parser.parse_args()

    try:
        command_instance = COMMANDS.get(args.command)
        if command_instance is None:
            raise ValueError(f"Command {args.command} is invalid")
        command_instance.execute(args)
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    main()
