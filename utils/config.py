
from datetime import datetime
from omegaconf import OmegaConf
import os


### Compose folders name from exp parameters
def compose_config_folders(config):

    prefix = config.checkpointing.prefix
    timestamp = str(datetime.timestamp(datetime.now()))

    exp_folder = config.checkpointing.checkpoint_path
    exp_name   = prefix + '_' + timestamp

    config.checkpointing.checkpoint_path = os.path.join(exp_folder, exp_name)


### Copy configuration to experiment folder
def copy_config_to_experiment_folder(config):

    folder = config.checkpointing.checkpoint_path
    if not os.path.exists(folder):
        os.mkdir(folder)

    out_path = os.path.join(folder, 'config.yaml')
    OmegaConf.save(config, out_path, resolve=True)

