
from utils import save_model

from pytorch_lightning import Trainer

from mains import Initialization


def main() -> None:

    model = Initialization()

    trainer = Trainer(gpus=1, max_epochs=1)
    trainer.fit(model)

    # save initialized map map as sample for inter surface map
    save_model('inits', model.net)


if __name__ == '__main__':
    main()
