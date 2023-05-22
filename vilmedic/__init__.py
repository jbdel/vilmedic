import logging

logging.getLogger('faiss.loader').setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("stanza").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.CRITICAL)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

from .zoo.modeling_auto import AutoModel
from .models import *
