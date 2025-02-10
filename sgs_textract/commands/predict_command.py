from utils.logger import logger
from commands.base_command import BaseCommand


class PredictCommand(BaseCommand):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--data", required=True)
        parser.add_argument("--output", required=True)

    @staticmethod
    def execute(args):
        logger.debug("Execute Predict Command")
