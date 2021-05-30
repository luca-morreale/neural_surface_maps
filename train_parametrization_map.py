
import torch
import hydra
from omegaconf import DictConfig

from utils import compose_config_folders
from utils import copy_config_to_experiment_folder
from utils import save_model

from pytorch_lightning import Trainer

from mains import ParametrizationMap



@hydra.main(config_path='experiments', config_name='parametrization_map')
def main(cfg: DictConfig) -> None:
    compose_config_folders(cfg)
    copy_config_to_experiment_folder(cfg)

    model = ParametrizationMap(cfg)
    model.net.load_state_dict(torch.load('inits/softplus_128_identity.pth'))

    trainer = Trainer(gpus=1, max_epochs=1)
    trainer.fit(model)

    # save surface map as sample for inter surface map
    save_model(cfg.checkpointing.checkpoint_path, model.net)



if __name__ == '__main__':
    main()
