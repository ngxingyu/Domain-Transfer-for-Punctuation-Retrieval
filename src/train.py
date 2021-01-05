import numpy as np
from transformers import AdamW
from torchcontrib.optim import SWA
from transformers import get_linear_schedule_with_warmup
from .utils.logger import get_logger
import subprocess
import datetime
import torch
from .config import *
from .engine import *

start_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
logger=get_logger(config.MODEL_NAME,start_time=start_time)
logger.info(vars(config))
logger.info(subprocess.check_output(['git', 'describe', '--always']))
logger.warning(start_time)
logger.info('cuda available: {}'.format(torch.cuda.is_available()))
device = torch.device(config.GPU_DEVICE if torch.cuda.is_available() else 'cpu')
logger.info('using device: {}'.format(device))
train_dataset=torch.load(config.TRAIN_DATASET)
dev_dataset=torch.load(config.DEV_DATASET)
train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)
dev_dataloader=torch.utils.data.DataLoader(dev_dataset, batch_size=config.DEV_BATCH_SIZE, num_workers=2)

model = BertCRFModel(num_punct=10, embedding_dim=config.EMBEDDING_DIM, hidden_dim=config.HIDDEN_DIM, use_crf=config.USE_CRF, logger=logger)

for i,param in enumerate(model.bert.parameters()):
    param.requires_grad = False
logger.info('load model')
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay
            )
        ],
        "weight_decay": 0.001,
    },
    {
        "params": [
            p for n, p in param_optimizer if any(
                nd in n for nd in no_decay
            )
        ],
        "weight_decay": 0.0,
    },
]

num_train_steps = train_dataset.tensors[0].size()[0] / config.TRAIN_BATCH_SIZE * config.FREEZE_EPOCHS

base_opt = AdamW(optimizer_parameters, lr=config.FREEZE_LEARNING_RATE)
optimizer = SWA(base_opt)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps
)

best_loss = np.inf
for epoch in range(config.FREEZE_EPOCHS):
    train_loss = train_fn(
        train_dataloader,
        model,
        optimizer,
        device,
        scheduler
    )
    optimizer.update_swa()
    optimizer.swap_swa_sgd()
    test_loss = eval_fn(
        dev_dataloader,
        model,
        device
    )
    optimizer.swap_swa_sgd()
    logger.info(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
    if test_loss < best_loss:
        torch.save(model.state_dict(), config.MODEL_PATH+config.MODEL_NAME+'-'+start_time+'.bin')
        best_loss = test_loss

model.load_state_dict(torch.load(config.MODEL_PATH+config.MODEL_NAME+'-'+start_time+'.bin'),strict=False)

for i,param in enumerate(model.bert.parameters()):
    if i<198-config.UNFROZEN_LAYERS:
        param.requires_grad = False
    else:
        param.requires_grad = True
        
num_train_steps = train_dataset.tensors[0].size()[0] / config.TRAIN_BATCH_SIZE * config.UNFREEZE_EPOCHS

base_opt = AdamW(optimizer_parameters, lr=config.UNFREEZE_LEARNING_RATE)
optimizer = SWA(base_opt)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps
)
        
for epoch in range(config.UNFREEZE_EPOCHS):
    train_loss = train_fn(
        train_dataloader,
        model,
        optimizer,
        device,
        scheduler
    )
    optimizer.update_swa()
    optimizer.swap_swa_sgd()
    test_loss = eval_fn(
        dev_dataloader,
        model,
        device
    )
    optimizer.swap_swa_sgd()
    logger.info(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
    if test_loss < best_loss:
        torch.save(model.state_dict(), config.MODEL_PATH+config.MODEL_NAME+'-'+start_time+'.bin')
        best_loss = test_loss
