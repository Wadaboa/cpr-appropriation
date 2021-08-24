import random
import os

import wandb
import torch
import numpy as np


IGNORE_INDEX = -100


def init_wandb(config):
    """
    Return a new W&B run to be used for logging purposes
    """
    assert isinstance(config, dict), "The given W&B config should be a dictionary"
    assert "api_key" in config, "Missing API key value in W&B config"
    assert "project" in config, "Missing project name in W&B config"
    assert "entity" in config, "Missing entity name in W&B config"
    assert "group" in config, "Missing group name in W&B config"

    os.environ["WANDB_API_KEY"] = config["api_key"]
    exclude_keys = ["api_key", "project", "entity", "group"]
    remaining_config = {k: v for k, v in config.items() if k not in exclude_keys}
    return wandb.init(
        project=config["project"],
        entity=config["entity"],
        group=config["group"],
        config=remaining_config,
    )


def set_seed(seed):
    """
    Fix all possible sources of randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
