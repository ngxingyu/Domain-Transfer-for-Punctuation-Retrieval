#%%
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from models import PunctuationDomainModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

# from icecream import install
# install()


# @hydra.main(config_name="core/config/config.yaml")
@hydra_runner(config_path="core/config", config_name="config")
def main(cfg : DictConfig) -> None:
    trainer=pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    do_training = True
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    model = PunctuationDomainModel(cfg.model, trainer=trainer)
    if do_training:
        trainer.fit(model)
        if cfg.model.nemo_path:
            model.save_to(cfg.model.nemo_path)
    gpu = 1 if cfg.trainer.gpus != 0 else 0
    trainer = pl.Trainer(gpus=gpu)
    model.set_trainer(trainer)
    queries = [
        'we bought four shirts one pen and a mug from the nvidia gear store in santa clara',
        'what can i do for you today',
        'how are you',
    ]
    inference_results = model.add_punctuation_capitalization(queries)

    for query, result in zip(queries, inference_results):
        logging.info(f'Query : {query}')
        logging.info(f'Result: {result.strip()}\n')

if __name__ == "__main__":
    main()