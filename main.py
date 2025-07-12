import os

os.environ.setdefault("COMET_DISABLE_AUTO_LOGGING", "1")
os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "1")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")

from contextvit.config import args
from contextvit.training import start_training, set_inductor_config

set_inductor_config()
start_training(args)
