#%%
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from data import PunctuationDataModule, PunctuationInferenceDataset
import os
from models import PunctuationDomainModel

# from nemo.core.config import hydra_runner
# from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from time import time
from pytorch_lightning.callbacks import ModelCheckpoint

import atexit
from copy import deepcopy
import snoop
snoop.install()

@hydra.main(config_path="../Punctuation_with_Domain_discriminator/2021-02-17_11-49-25/",config_name="hparams.yaml")
def main(cfg : DictConfig) -> None:
    torch.set_printoptions(sci_mode=False)
    # trainer=pl.Trainer(**cfg.trainer)
    # exp_manager(trainer, cfg.get("exp_manager", None))
    # do_training = False
    # logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    # if do_training:
    #     trainer.fit(model)
    #     if cfg.model.nemo_path:
    #         model.save_to(cfg.model.nemo_path)
    # gpu = 1 if cfg.trainer.gpus != 0 else 0
    # model = PunctuationDomainModel.restore_from(restore_path=cfg.exp_manager.restore_path, override_config_path=cfg.exp_manager.override_config_path, )
    model = PunctuationDomainModel.load_from_checkpoint( #TEDend2021-02-11_07-57-33  # TEDstart2021-02-11_07-55-58
    checkpoint_path="/home/nxingyu2/project/Punctuation_with_Domain_discriminator/2021-02-17_11-49-25/checkpoints/Punctuation_with_Domain_discriminator-last.ckpt")
    trainer = pl.Trainer(**cfg.trainer)
    # trainer = pl.Trainer(gpus=gpu)
    trainer.test(model,ckpt_path=None)
    # queries = [
    #     'we bought four shirts one pen and a mug from the nvidia gear store in santa clara',
    #     'what can i do for you today',
    #     'how are you',
    # ]
    # inference_results = model.add_punctuation_capitalization(queries)

    # for query, result in zip(queries, inference_results):
    #     logging.info(f'Query : {query}')
    #     logging.info(f'Result: {result.strip()}\n')

if __name__ == "__main__":
    main()