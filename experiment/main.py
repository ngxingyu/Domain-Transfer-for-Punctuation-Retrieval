#%%
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from icecream import ic, install
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from data import PunctuationDataModule, PunctuationInferenceDataset

# from models import PunctuationDomainModel

# from nemo.core.config import hydra_runner
# from nemo.utils import logging
# from nemo.utils.exp_manager import exp_manager

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
    
    ic(cfg)
    cfg.model.punct_label_ids=OmegaConf.create(sorted(cfg.model.punct_label_ids))
    labels_to_ids = {_[1]:_[0] for _ in enumerate(cfg.model.punct_label_ids)}
    ids_to_labels = {_[0]:_[1] for _ in enumerate(cfg.model.punct_label_ids)}def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
'''
    dm=PunctuationDataModule(
            tokenizer= cfg.model.transformer_path,
            labelled= list(cfg.model.dataset.labelled),
            unlabelled= list(cfg.model.dataset.unlabelled),
            punct_label_ids= labels_to_ids,
            train_batch_size= cfg.model.train_ds.batch_size,
            max_seq_length= cfg.model.dataset.max_seq_length,
            val_batch_size= cfg.model.validation_ds.batch_size,
            num_workers= cfg.model.dataset.num_workers,
            pin_memory= cfg.model.dataset.pin_memory,
            train_shuffle= cfg.model.train_ds.shuffle,
            val_shuffle= cfg.model.validation_ds.shuffle,
    )
    dm.setup('fit')
    dl=dm.train_dataloader()
    ic(next(iter(dl)))
    # ic(next(iter(dm.train_dataloader())))
    


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
