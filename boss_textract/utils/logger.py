from loguru import logger
from tqdm import tqdm
from pathlib import Path


class _Logger:
    _instance = None

    def __new__(cls, *args, **kawrgs):
        if not cls._instance:
            cls._instance = logger
            cls._instance.configure(
                handlers=[
                    dict(
                        sink=lambda msg: tqdm.write(msg, end=""),
                        format="<green>{time:HH:mm:ss.SSS}</green> [<level>{level: ^8}</level>] <level>{message}</level>",
                        colorize=True,
                    )
                ]
            )
        return cls._instance


log_path = Path(__file__).absolute().parents[2] / "logs/"
logger = _Logger()
