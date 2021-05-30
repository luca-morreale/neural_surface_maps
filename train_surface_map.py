
import hydra
from omegaconf import DictConfig

from utils import compose_config_folders
from utils import copy_config_to_experiment_folder
from utils import save_meta_sample

from pytorch_lightning import Trainer

from mains import SurfaceMap



@hydra.main(config_path='experiments', config_name='surface_map')
def main(cfg: DictConfig) -> None:
    compose_config_folders(cfg)
    copy_config_to_experiment_folder(cfg)

    model = SurfaceMap(cfg)

    trainer = Trainer(gpus=1, max_epochs=1)
    trainer.fit(model)

    # save surface map as sample for inter surface map
    save_meta_sample(cfg.checkpointing.checkpoint_path, model.dataset.sample, model.net)



if __name__ == '__main__':
    main()
