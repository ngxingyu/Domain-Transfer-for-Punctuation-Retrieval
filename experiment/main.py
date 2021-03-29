
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from data import PunctuationInferenceDataset
import os
from models import PunctuationDomainModel

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
        pp(os.system(f'rm -r {cfg.tmp_path}/*.{data_id}*'))
    atexit.register(savecounter)

    cfg.model.maximum_unfrozen=max(cfg.model.maximum_unfrozen,cfg.model.unfrozen)
    pl.seed_everything(cfg.seed)
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
<<<<<<< HEAD
        patience=6,
=======
        patience=2,
>>>>>>> 685fc40118c0a5b1039c9fc2926f4bd42aa03d13
        verbose=False,
        mode='max'
    )
    trainer = pl.Trainer(callbacks=[early_stop_callback],**cfg.trainer) #,track_grad_norm=2
    log_dir=exp_manager(trainer, cfg.exp_manager).__str__()
    model = PunctuationDomainModel(cfg=cfg, trainer=trainer, data_id = data_id,log_dir=log_dir)

    lrs=[1e-2,1e-5] if cfg.model.frozen_lr is None else list(cfg.model.frozen_lr)
    while(model.hparams.model.unfrozen<=cfg.model.maximum_unfrozen and model.hparams.model.unfrozen>=0):
        model.hparams.model.optim.lr = lrs.pop(0)
<<<<<<< HEAD
        # model.hparams.model.domain_head.gamma=gamma.pop(0)
=======
>>>>>>> 685fc40118c0a5b1039c9fc2926f4bd42aa03d13
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

if __name__ == "__main__":
    main()

# %%
