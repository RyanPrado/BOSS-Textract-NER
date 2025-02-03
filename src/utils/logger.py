from loguru import logger
from tqdm import tqdm

logger_format = "<green>{time:HH:mm:ss.SSS}</green> [<level>{level: ^8}</level>] <level>{message}</level>"
logger.configure(
    handlers=[
        dict(
            sink=lambda msg: tqdm.write(msg, end=""),
            format=logger_format,
            colorize=True,
        )
    ]
)

logger = logger
