import yaml
from contextvit.config import set_config
set_config()
from contextvit.training import start_training


with open("params.yaml", "r") as f:
    args = yaml.safe_load(f)

start_training(args)
