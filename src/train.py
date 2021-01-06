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
logger=get_logger(config.model_name,start_time=start_time)
logger.info(vars(config))
logger.info(subprocess.check_output(['git', 'describe', '--always']))
logger.warning(start_time)
logger.info('cuda available: {}'.format(torch.cuda.is_available()))
device = torch.device(config.gpu_device if torch.cuda.is_available() else 'cpu')
logger.info('using device: {}'.format(device))
train_dataset=torch.load(config.train_dataset)
dev_dataset=torch.load(config.dev_dataset)
train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, num_workers=4)
dev_dataloader=torch.utils.data.DataLoader(dev_dataset, batch_size=config.dev_batch_size, num_workers=2)

model = BertCRFModel(num_punct=10, embedding_dim=config.embedding_dim, hidden_dim=config.hidden_dim, use_crf=config.use_crf, logger=logger)

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

num_train_steps = train_dataset.tensors[0].size()[0] / config.train_batch_size * config.freeze_epochs

base_opt = AdamW(optimizer_parameters, lr=config.freeze_lr)
optimizer = SWA(base_opt)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps
)

best_loss = np.inf
for epoch in range(config.freeze_epochs):
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
        torch.save(model.state_dict(), config.model_path+config.model_name+'-'+start_time+'.bin')
        best_loss = test_loss

model.load_state_dict(torch.load(config.model_path+config.model_name+'-'+start_time+'.bin'),strict=False)

for i,param in enumerate(model.bert.parameters()):
    if i<198-config.unfreeze_layers:
        param.requires_grad = False
    else:
        param.requires_grad = True

num_train_steps = train_dataset.tensors[0].size()[0] / config.train_batch_size * config.unfreeze_epochs

base_opt = AdamW(optimizer_parameters, lr=config.unfreeze_lr)
optimizer = SWA(base_opt)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps
)

for epoch in range(config.unfreeze_epochs):
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
        torch.save(model.state_dict(), config.model_path+config.model_name+'-'+start_time+'.bin')
        best_loss = test_loss

