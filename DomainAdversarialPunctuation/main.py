#%%
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from models import PunctuationDomainModel
# from core.config import hydra_runner
from utils import logging
from icecream import install
install()

@hydra.main(config_name="core/config/config.yaml")
def main(cfg : DictConfig) -> None:
    # trainer=pl.Trainer(**cfg.trainer)
    # print(cfg)
    # print(OmegaConf.to_yaml(cfg))
    ic(cfg.model.class_labels)

if __name__ == "__main__":
    main()

from hydra.experimental import initialize, compose
with initialize(config_path="core/config/"):
    cfg=compose(config_name="config")
## %%
from data import PunctuationDataModule
dm=PunctuationDataModule(
    list(cfg.model.dataset.labelled),
    list(cfg.model.dataset.unlabelled),
    cfg.model.train_ds.batch_size,
    cfg.model.dataset.max_seq_length,
    cfg.model.validation_ds.batch_size,
    cfg.model.dataset.num_workers,
    cfg.model.dataset.pin_memory,
    cfg.model.dataset.drop_last,
    )
dm.setup('fit')
#%%
#scheduler test

