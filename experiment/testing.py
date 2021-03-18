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

# exp='2021-03-17_15-42-41'
# exp='2021-03-17_15-44-24'
# exp='2021-03-17_15-54-07'
exp='2021-03-18_09-18-35'

@hydra.main(config_path=f"../Punctuation_with_Domain_discriminator/{exp}/",config_name="hparams.yaml")
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
    checkpoint_path=f"/home/nxingyu2/project/Punctuation_with_Domain_discriminator/{exp}/checkpoints/Punctuation_with_Domain_discriminator-last.ckpt")
    # checkpoint_path=f"/home/nxingyu2/project/Punctuation_with_Domain_discriminator/{exp}/checkpoints/Punctuation_with_Domain_discriminator---val_loss=0.34-epoch=7.ckpt")
    
    model.dm.test_dataset=PunctuationDomainDatasets(split='test',
                    # num_samples=model.dm.val_batch_size,
                    num_samples=8,
                    max_seq_length=model.dm.max_seq_length,
                    punct_label_ids=model.dm.punct_label_ids,
                    label_map=model.dm.label_map,
                    # labelled=['/home/nxingyu2/data/ted2010'],
                    labelled=['/home/nxingyu2/data/switchboardutt_processed'],
                    # labelled=['/home/nxingyu2/data/open_subtitles_processed'],
                    # labelled=['/home/nxingyu2/data/ted_talks_processed'], #jointteduttdice32acc4bs16
                    unlabelled=[],
                    tokenizer=model.dm.tokenizer,
                    randomize=model.dm.val_shuffle,
                    data_id=model.dm.data_id,
                    tmp_path=model.dm.tmp_path,
                    attach_label_to_end=model.dm.attach_label_to_end,
                    no_space_label=model.dm.no_space_label,
                    pad_start=model.dm.pad_start,
                    )
    model.hparams.log_dir=f"/home/nxingyu2/project/Punctuation_with_Domain_discriminator/{exp}/"
    trainer = pl.Trainer(**cfg.trainer)
    trainer.test(model,ckpt_path=None)

    queries = [
        "Okay, Ellen what kind of a car do you think you're going to buy?. Well, as a matter of fact, was thinking about that the other day, and, uh, really don't know the answer, uh, would sort of like to, uh, think about something in the way of, uh, uh, sort of a sporty car but not any, not, you know, a luxury type sporty one. Yeah. But, um, something that still has a lots of amenities and, you know, gadgets and things. Oh, you do want a lot of that stuff? Yeah, well, yeah like, like some of those things. They come in really handy . What kind of, uh, things are you going to consider, you know, what, uh, you said something about the, about the, well, what do you call them, you said amenities, Amenities. that they have, but what about, um, their reputation of the company or the price. Yeah, well, of course, guess, uh, price is always the big consideration, but, It is for me, other people, yeah. don't seem to have the same problem . Well, that's, that's a big one in my book, Yeah. but, uh, um, have preferences for, uh, for some, uh, makers over others, um, and would sort of like to buy American, Yeah. but, you know, I'm not so totally hung up on that, that wouldn't buy something else, how about you? Well, um, the last car we bought was American because of, because of that reason, but have not been entirely happy with, uh, several things about the car, it doesn't seem like the quality is quite as high as expected it to be. Oh, really? Because several things, minor things sort of, but still they cost us money, um, that we didn't feel like we should have had to pay, on a car that, that was that new, Uh-huh. you know, we bought the car new and after, um, well, well, well under two years we had to replace the clutch. Oh. And, they just said, well, you know, clutches are disposable , and said, since when? Yeah. Brake pads are disposable , Yeah. you know, we know that, but never thought a clutch was disposable. Yeah, wouldn't have thought so either. Yeah, so that was, that was kind of a shock . Oh. Yeah, I, guess there's a lot to, to think about when you're trying to make that decision. Yeah, you know, the less actually, the less you spend on a car it seems like luxury cars, they're called luxury cars even though they're much more expensive like, like, uh, uh, a Mercedes Benz, they don't have the history of breaking down or things like that, that would go wrong would definitely not be considered disposable. Right. You would never think of having to replace the clutch in a Mercedes, No but then, especially not after two years.",
    # 'we bought four shirts one pen and a mug from the nvidia gear store in santa clara',
    # 'what can i do for you today',
    # 'how are you',
    # ' in guadeloupe or marti nique it also brin gs into ques tion budgetary po licy because the europe an union is after all making a present of ecu 1.9 billion to three multinationals where are the financial interests of the european union firstly development policy in africa in any case in the acp countries employ ment policy in madeira the canaries guadel oupe martinique and crete regional pol icy in the ultra-peripheral areas human rights which mr barthet mayer mentioned earlier since dollar bananas are after all slavery bana nas the product of human exploitation by three multinat ionals payments of ecu 50 per mon th instead of ecu 50 per day',
    # '''Plans for this weekend include turning wine into water. The small white buoys marked the location of hundreds of crab pots. He said he was not there yesterday; however, many people saw him there. Today arrived with a crash of my car through the garage door. The lyrics of the song sounded like fingernails on a chalkboard. The Guinea fowl flies through the air with all the grace of a turtle. They ran around the corner to find that they had traveled back in time.''',
    # 'good morning everyone how have your weekends been its a really great day thank you',
    # 'first of all i too agree that tourism related action must include employment training and education as you know after the european conference on tourism and employment in luxemborg we set up a high-level group whose mission was to examine how best tourism could contribute towards employment the first stage',
    # "Ummm Im not really sure can you check with mr bob instead thanks",
    # "The following are from the switchboard corpus. Okay. Well, discussing air pollution today, guess. Uh-huh. Um, uh, well, give me your first impressions. Uh, don't know, there's a lot of air pollution. Yeah. Um, think industries and companies provide a lot of it … Uh-huh. … and, with, uh, guess with the oil burning over in Kuwait and stuff, that would have a lot of air pollution in it. Uh, puts a whole, yeah, gets a whole new picture to what real air pollution can be, but, uh, that stuff going on over there. What, what, uh, what part of Pennsylvania are you in? Um, I'm north of Pittsburgh, so. ",
    # "Okay, okay, so, it's, it's amazing too, you know, with that, with the oil wells burning over there, that's the exact same stuff that's coming out of cars every day, just in, uh, just in a little different grade, guess. Yeah. But, uh, in Dallas, we've got, we've just, uh, brought in a whole new set of requirements on inspections and things like that for cars, because, uh, people just don't use mass transit and stuff in Dallas. Everybody loves their car, and you see an awful lot of, uh, one person vehicles on the road on, during rush hours. That seems to be our biggest problem down here. Um, you know, there's, uh, there's a lot of industry around, but, uh, it's not, it's - any pollution that industry's dumping around here is not going into the air. It's typically water type situation. ",
    # "Yeah. Um. We have a couple, we have like a steel mill and a couple refineries and stuff, and know there's a lot of air pollution going in there, and like they, they get fined whenever they do the air pollution … Uh-huh. … but the fine is nothing, you know, Yeah, it's, it's, it's like nothing to them. like, like two hours of output or something like that. Yeah, yeah, That's true, that's true. Uh-huh. Yeah, I, a, uh, grew up in South Dakota, so was never, was never exposed … Uh-huh. … to anything of, of the, of the sort. Um, there were always P people and what not were always telling us that, uh, farm chemicals and what not were destroying our water system and all that … Uh-huh. … but we just, we just never saw the results. There was, there was dust in the air during planting seasons and what not, but that, that was all we ever saw, and then five years ago moved to Dallas, and suddenly started to understand what burning eyes, and all that stuff is about that I'd always heard about. ",
    # "It, uh, it's, it gets, it's real depressing. In the morning sometime you can tell if it's a good day or a bad day by, uh, how far out from downtown, uh, you can be on the road and still not see it. Oh, my God . And, uh, yeah, I mean, it's not, don't think Dallas is considered, uh, a real bad place for air pollution, but, you, you can tell, you can tell the differences in the days, when it's, when the haze is kind of yellowish gray instead of just being a, a foggy, misty color … Uh-huh. … and, uh, it, it's a little, it's a little disappointing sometimes you start to realize what you're breathing. ",
    # "Uh, yeah, but, don't know what they can do to really prevent it, you know. Like, how, what can they do about the oil burning over in Kuwait? What, you know, I mean they fine the industries, but you know, that doesn't seem to stop them there … Right. … don't know what else they can do. Right. It, it's that there really isn't a whole lot. It's one of those, uh, it's one of those things that if they do a little bit and, uh, and, you know, e-, every little bit does help, do believe that. Um, but also believe that the earth is a kind of a self-regulating system, and, uh, it will clean itself up, eventually. It, the whole idea is not to, not to push the limit too hard, guess … Uh-huh. … let the, you know, let the natural, natural systems take care of the problem as much as possible. Yeah, I, I, yeah, just don't understand, you know, what else anyone can do about it. don't think it's something that people really think about, either. Uh-huh. You know, it's, I mean, it, it should probably be a big issue, you know, because it's, it's doing a lot of damage, but I, it's something, you know, don't think many people really think about it, because it's nothing they, don't think we really have too much control over it. Right. And it's one of those things, it, it's so hard to measure what, what the damage is … Uh-huh. … it's kind of like, oh, guess it's kind of like, kind of like cigarette smoking, you know. It, it could go on for years and years until they start to see some results and people can actually, actually say, Yeah, it's, it's, it's doing, doing some damage … Uh-huh. … and something's got to be done. ",
    # "Um, there's, you know, there's a lot of things like that. It, you can, uh, you can pound on something for a long time before it finally breaks, but until it breaks, you don't really know that there, you were doing anything to it. Yeah, exactly. But, uh, Well, really don't know too much else about it. Yeah, well, that's, that's, think we both agree it's, it's one of those deals that, uh, just think there's a lot of other problems right now … Uh-huh. … and, uh, we've done a lot to take care of it. Yeah. And, uh, Yeah, we have tried, I mean, you know, and, um, I, know where, you know, where a couple of the mills that have, know they put things on their stacks, you know, to filter the smoke, and do all kinds of things, but, I mean, every now and then it breaks, you know, and … Oh, sure. … and you just have smoke going out into the air for a day or two, until you can get it fixed. Sure. and so, so, you know, it's something we, we have tried to help. Oh, there's no doubt about it. ",
    # "Yeah. don't know if you ever happened to see some of the like, Twenty Twenty and what not about Rumania and East Germany when they first got pictures out of there … Uh-huh. … about how some of their systems had been running for twenty and thirty years … Uh-huh. … and, uh, you know, they had absolutely no regulations, no controls whatsoever, and they had destroyed entire forests and what not, just because the air was so polluted. That's, that's the kind of things that, uh, you don't see in this country, and that's, that's why think that, you know, it's, don't know if you can ever do enough, but, uh, think it's all relative to the, to the time and place, and think right now it's, it's pretty much under control. Yeah, okay, well. All righty. Uh-huh. It's been nice talking to you. Well, you bet. Okay, bye-bye. Bye.",
    "firstly development policy in africa in any case in the acp countries employment policy in madeira the canaries guadeloupe martinique and crete regional policy in the ultra-peripheral areas human rights which mr barthet-mayer mentioned earlier since dollar bananas are after all slavery bananas the product of human exploitation by three multinationals payments of ecu 50 per month instead of ecu 50 per day in guadeloupe or martinique it also brings into question budgetary policy because the european union is after all making a present of ecu 1.9 billion to three multinationals where are the financial interests of the european union",
]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_results = model.to(device).add_punctuation(queries)
    for query, result in zip(queries, inference_results):
        print(f'Query : {query}')
        print(f'Result: {result.strip()}\n\n')

if __name__ == "__main__":
    main()