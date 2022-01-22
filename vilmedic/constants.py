import os
from appdirs import user_cache_dir

DATA_DIR = __file__.replace('vilmedic/constants.py', 'data')

CACHE_DIR = user_cache_dir("vilmedic")
EXTRA_CACHE_DIR = os.path.join(CACHE_DIR, "extras")
MODEL_ZOO_CACHE_DIR = os.path.join(CACHE_DIR, "zoo", "models")