# Copyright (c) 2021 <Ng Xing Yu> 

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from data import PunctuationDataModule, PunctuationInferenceDataset, PunctuationDomainDatasets
import os
from models import PunctuationDomainModel

from nemo.utils.exp_manager import exp_manager
from time import time
from pytorch_lightning.callbacks import ModelCheckpoint

import atexit
from copy import deepcopy
import snoop
snoop.install()

exp='results/2021-03-27_18-00-46'
exp='2021-03-28_09-09-50'
exp='2021-03-28_11-56-37'

@hydra.main(config_path=f"../Punctuation_with_Domain_discriminator/{exp}/",config_name="hparams.yaml")
# @hydra.main(config_name="config.yaml")
def main(cfg : DictConfig) -> None:
    pl.seed_everything(cfg.seed)
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

<<<<<<< HEAD
    model = PunctuationDomainModel.load_from_checkpoint( #TEDend2021-02-11_07-57-33  # TEDstart2021-02-11_07-55-58
    checkpoint_path=f"/home/nxingyu/project/Punctuation_with_Domain_discriminator/{exp}/checkpoints/Punctuation_with_Domain_discriminator-last.ckpt")
    model._cfg.model.dataset.labelled=['/home/nxingyu/data/switchboardutt_processed']
    model._cfg.model.dataset.unlabelled=[]
    model.setup_datamodule()

    model.hparams.log_dir=f"/home/nxingyu/project/Punctuation_with_Domain_discriminator/{exp}/"
    trainer = pl.Trainer(**cfg.trainer)
    trainer.test(model,ckpt_path=None)

=======
    model = PunctuationDomainModel.load_from_checkpoint(
    checkpoint_path=f"/home/nxingyu2/project/Punctuation_with_Domain_discriminator/{exp}/checkpoints/Punctuation_with_Domain_discriminator-last.ckpt")
    
    # model._cfg.model.dataset.labelled=['/home/nxingyu2/data/switchboardutt_processed']
    # model._cfg.model.dataset.labelled=['/home/nxingyu2/data/ted_talks_processed']
    # model._cfg.model.dataset.labelled=['/home/nxingyu2/data/open_subtitles_processed']
    # model._cfg.model.dataset.labelled=['/home/nxingyu2/data/lrec_processed']
    # model._cfg.model.dataset.labelled=['/home/nxingyu2/data/ted2010_processed']

    model._cfg.model.dataset.unlabelled=[]
    # model._cfg.model.test_chunk_percent=0.5

    # model.setup_datamodule()
    # model.hparams.log_dir=f"/home/nxingyu2/project/Punctuation_with_Domain_discriminator/{exp}/"
    # trainer = pl.Trainer(**cfg.trainer)
    # trainer.test(model,ckpt_path=None)

>>>>>>> 685fc40118c0a5b1039c9fc2926f4bd42aa03d13
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # inference_results = model.to(device).add_punctuation(queries)
    # for query, result in zip(queries, inference_results):
    #     print(f'Query : {query}\n')
    #     print(f'Result: {result.strip()}\n\n')

<<<<<<< HEAD
    # while 1:
    #     text=input('Enter text to punctuate:\n')
    #     texts=[text]
    #     inference_results = model.to(device).add_punctuation(texts)
    #     for text, result in zip(texts, inference_results):
    #         print(f'\n\nQuery : {text}\n')
    #         print(f'Result: {result.strip()}\n\n')
=======
    import pandas as pd
    sample=pd.read_csv('/home/nxingyu2/data/switchboardutt_processed.test.csv').itertuples()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    while 1:
        x=input('Press Enter to loop through samples or insert your own texts\n--[ ')
        if x=='':
            texts=[next(sample)[2]]
            inference_results = model.to(device).add_punctuation(texts)
            for text, result in zip(texts, inference_results):
                print(f'\n\nQuery : {text}\n')
                print(f'Result: {result.strip()}\n\n')
        else:
            texts=[x]
            inference_results = model.to(device).add_punctuation(texts)
            for text, result in zip(texts, inference_results):
                print(f'\n\nQuery : {text}\n')
                print(f'Result: {result.strip()}\n\n')
>>>>>>> 685fc40118c0a5b1039c9fc2926f4bd42aa03d13


queries = [
        "Okay, Ellen what kind of a car do you think you're going to buy?. Well, as a matter of fact, was thinking about that the other day, and, uh, really don't know the answer, uh, would sort of like to, uh, think about something in the way of, uh, uh, sort of a sporty car but not any, not, you know, a luxury type sporty one. Yeah. But, um, something that still has a lots of amenities and, you know, gadgets and things. Oh, you do want a lot of that stuff? Yeah, well, yeah like, like some of those things. They come in really handy . What kind of, uh, things are you going to consider, you know, what, uh, you said something about the, about the, well, what do you call them, you said amenities, Amenities. that they have, but what about, um, their reputation of the company or the price. Yeah, well, of course, guess, uh, price is always the big consideration, but, It is for me, other people, yeah. don't seem to have the same problem . Well, that's, that's a big one in my book, ",
        "Yeah. but, uh, um, have preferences for, uh, for some, uh, makers over others, um, and would sort of like to buy American, Yeah. but, you know, I'm not so totally hung up on that, that wouldn't buy something else, how about you? Well, um, the last car we bought was American because of, because of that reason, but have not been entirely happy with, uh, several things about the car, it doesn't seem like the quality is quite as high as expected it to be. Oh, really? Because several things, minor things sort of, but still they cost us money, um, that we didn't feel like we should have had to pay, on a car that, that was that new, Uh-huh. you know, we bought the car new and after, um, well, well, well under two years we had to replace the clutch. Oh. And, they just said, well, you know, clutches are disposable , and said, since when? Yeah. Brake pads are disposable , Yeah. you know, we know that, but never thought a clutch was disposable. Yeah, wouldn't have thought so either. Yeah, so that was, that was kind of a shock . Oh. Yeah, I, guess there's a lot to, to think about when you're trying to make that decision. Yeah, you know, the less actually, the less you spend on a car it seems like luxury cars, they're called luxury cars even though they're much more expensive like, like, uh, uh, a Mercedes Benz, they don't have the history of breaking down or things like that, that would go wrong would definitely not be considered disposable. Right. You would never think of having to replace the clutch in a Mercedes, No but then, especially not after two years.",
    'we bought four shirts one pen and a mug from the nvidia gear store in santa clara',
    'what can i do for you today',
    'how are you',
    ' in guadeloupe or marti nique it also brin gs into ques tion budgetary po licy because the europe an union is after all making a present of ecu 1.9 billion to three multinationals where are the financial interests of the european union firstly development policy in africa in any case in the acp countries employ ment policy in madeira the canaries guadel oupe martinique and crete regional pol icy in the ultra-peripheral areas human rights which mr barthet mayer mentioned earlier since dollar bananas are after all slavery bana nas the product of human exploitation by three multinat ionals payments of ecu 50 per mon th instead of ecu 50 per day',
    '''Plans for this weekend include turning wine into water. The small white buoys marked the location of hundreds of crab pots. He said he was not there yesterday; however, many people saw him there. Today arrived with a crash of my car through the garage door. The lyrics of the song sounded like fingernails on a chalkboard. The Guinea fowl flies through the air with all the grace of a turtle. They ran around the corner to find that they had traveled back in time.''',
    "Yeah. don't know if you ever happened to see some of the like, Twenty Twenty and what not about Rumania and East Germany when they first got pictures out of there … Uh-huh. … about how some of their systems had been running for twenty and thirty years … Uh-huh. … and, uh, you know, they had absolutely no regulations, no controls whatsoever, and they had destroyed entire forests and what not, just because the air was so polluted. That's, that's the kind of things that, uh, you don't see in this country, and that's, that's why think that, you know, it's, don't know if you can ever do enough, but, uh, think it's all relative to the, to the time and place, and think right now it's, it's pretty much under control. Yeah, okay, well. All righty. Uh-huh. It's been nice talking to you. Well, you bet. Okay, bye-bye. Bye.",
    "firstly development policy in africa in any case in the acp countries employment policy in madeira the canaries guadeloupe martinique and crete regional policy in the ultra-peripheral areas human rights which mr barthet-mayer mentioned earlier since dollar bananas are after all slavery bananas the product of human exploitation by three multinationals payments of ecu 50 per month instead of ecu 50 per day in guadeloupe or martinique it also brings into question budgetary policy because the european union is after all making a present of ecu 1 point 9 billion to three multinationals where are the financial interests of the european union",
    "Tolkien drew on a wide array of influences including language, Christianity, mythology including the Norse Völsunga saga, archaeology, especially at the Temple of Nodens, ancient and modern literature, and personal experience. He was inspired primarily by his profession, philology; his work centred on the study of Old English literature, especially Beowulf, and he acknowledged its importance to his writings.",
]

if __name__ == "__main__":
    main()
