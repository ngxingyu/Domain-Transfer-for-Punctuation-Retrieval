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

import atexit
from copy import deepcopy
import snoop
snoop.install()

@hydra.main(config_name="config")
def main(cfg: DictConfig)->None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_printoptions(sci_mode=False)
    data_id = str(int(time()))
    def savecounter():
        # pp(os.system(f'rm -r {cfg.model.dataset.data_dir}/*.{data_id}.csv'))
        pp(os.system(f'rm -r {cfg.tmp_path}/*.{data_id}*'))
    atexit.register(savecounter)

    cfg.model.maximum_unfrozen=max(cfg.model.maximum_unfrozen,cfg.model.unfrozen)

    pp(cfg)
    pl.seed_everything(cfg.seed)
    
    trainer = pl.Trainer(**cfg.trainer) #,track_grad_norm=2
    log_dir=exp_manager(trainer, cfg.exp_manager).__str__()
    model = PunctuationDomainModel(cfg=cfg, trainer=trainer, data_id = data_id,log_dir=log_dir)

    lrs=[1e-2,1e-5] if cfg.model.frozen_lr is None else list(cfg.model.frozen_lr)
    # gamma=[1e-1,1e-2] if cfg.model.gamma is None else list(cfg.model.gamma)
    while(model.hparams.model.unfrozen<=cfg.model.maximum_unfrozen and model.hparams.model.unfrozen>=0):
        # trainer.current_epoch=0
        # lr_finder = trainer.tuner.lr_find(model,min_lr=1e-8, max_lr=0.5, num_training=80) #, early_stop_threshold=None
        # # Results can be found in
        # pp(lr_finder.results)
        # new_lr = lr_finder.suggestion()
        # model.hparams.model.optim.lr = new_lr
        # model.dm.reset()
        model.hparams.model.optim.lr = lrs.pop(0)
        # model.hparams.model.domain_head.gamma=gamma.pop(0)
        trainer.current_epoch=0
        trainer.fit(model)
        try:
            model.unfreeze(cfg.model.unfreeze_step)
        except:
            pp('training complete.')
            break
    if cfg.model.nemo_path:
        model.save_to(cfg.model.nemo_path)

    
    
    gpu = 1 if cfg.trainer.gpus != 0 else 0
    # model.dm.setup('test')
    test_trainer = pl.Trainer(gpus=gpu)
    test_trainer.test(model,ckpt_path=None)
    


# @hydra.main(config_name="config.yaml")
# def main(cfg : DictConfig) -> None:
#     trainer=pl.Trainer(**cfg.trainer)
#     exp_manager(trainer, cfg.get("exp_manager", None))
#     do_training = True
#     logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
#     model = PunctuationDomainModel(cfg.model, trainer=trainer)
#     if do_training:
#         trainer.fit(model)
#         if cfg.model.nemo_path:
#             model.save_to(cfg.model.nemo_path)
#     gpu = 1 if cfg.trainer.gpus != 0 else 0
#     trainer = pl.Trainer(gpus=gpu)
#     model.set_trainer(trainer)
#     queries = [
#         'we bought four shirts one pen and a mug from the nvidia gear store in santa clara',
#         'what can i do for you today',
#         'how are you',
#     ]
#     inference_results = model.add_punctuation_capitalization(queries)

#     for query, result in zip(queries, inference_results):
#         logging.info(f'Query : {query}')
#         logging.info(f'Result: {result.strip()}\n')

if __name__ == "__main__":
    main()

# %%
