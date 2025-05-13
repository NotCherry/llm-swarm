from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
import logging

logging.getLogger("websockets").propagate = False
logging.getLogger("requests").propagate = False


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler('app.log', maxBytes=1000000, backupCount=5)
file_handler.setFormatter(file_format)
log = logging.getLogger("rich")
log.addHandler(file_handler)


SELECTED_MODEL = "meta-llama/Llama-3.2-1B-Instruct"