
import torch
import hydra
from omegaconf import DictConfig

from utils import compose_config_folders
from utils import copy_config_to_experiment_folder
from utils import save_model

from pytorch_lightning import Trainer

from mains import InterSurfaceMap



@hydra.main(config_path='experiments', config_name='inter_surface_map')
def main(cfg: DictConfig) -> None:
    compose_config_folders(cfg)
    copy_config_to_experiment_folder(cfg)

    model = InterSurfaceMap(cfg)
    model.net.load_state_dict(torch.load('inits/softplus_128_identity.pth'))

    trainer = Trainer(gpus=1, max_epochs=1)
    trainer.fit(model)

    # save surface map as sample for inter surface map
    save_model(cfg.checkpointing.checkpoint_path, model.net)
    # potentially you can save the rotation in your
    # map (model.net) so you don't have to recompute it



if __name__ == '__main__':
    main()
