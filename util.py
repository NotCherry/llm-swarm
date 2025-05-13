from rich.logging import RichHandler
import logging

logging.getLogger("websockets").propagate = False
logging.getLogger("requests").propagate = False


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")

SELECTED_MODEL = "meta-llama/Llama-3.2-1B-Instruct"