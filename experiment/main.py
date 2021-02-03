#%%
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from icecream import ic, install
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from data import PunctuationDataModule, PunctuationInferenceDataset
import os
from models import PunctuationDomainModel

# from nemo.core.config import hydra_runner
# from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from time import time
import matplotlib.pyplot as plt
import atexit

def toString(obj):
    if isinstance(obj, np.ndarray):
        return f'array shape {obj.shape.__str__()} type {obj.dtype}'
    if isinstance(obj, torch.Tensor):
        return f'tensor shape {obj.shape.__str__()} type {obj.dtype}'
    if isinstance(obj, dict):
        return {_[0]:toString(_[1]) for _ in obj.items()}.__str__()
    return repr(obj)
install()
ic.configureOutput(argToStringFunction=toString)

@hydra.main(config_name="config")
def main(cfg: DictConfig)->None:
    data_id = str(int(time()))
    def savecounter():
        ic(os.system(f'rm -r {cfg.model.dataset.data_dir}/*.{data_id}.csv'))
    atexit.register(savecounter)
    ic(cfg)
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)
    model = PunctuationDomainModel(cfg=cfg, trainer=trainer, data_id = data_id)
    model.setup_datamodule()
    
    lr_finder = trainer.tuner.lr_find(model,model.dm)
    ic(lr_finder.results)
    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

    new_lr = ic(lr_finder.suggestion())
    model.hparams.optim.lr = new_lr

    trainer.fit(model, model.dm)
    if cfg.model.nemo_path:
        model.save_to(cfg.model.nemo_path)
    
    gpu = 1 if cfg.trainer.gpus != 0 else 0
    # model.dm.setup('test')
    trainer = pl.Trainer(gpus=gpu)
    trainer.test(model,datamodule=model.dm,ckpt_path=None)


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
