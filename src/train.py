import numpy as np
from transformers import AdamW
from torchcontrib.optim import SWA
from transformers import get_linear_schedule_with_warmup

import torch
from config import *
from engine import *
from model import EntityModel

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
train_dataset=torch.load(config.TRAIN_DATASET)
dev_dataset=torch.load(config.DEV_DATASET)
train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)
dev_dataloader=torch.utils.data.DataLoader(dev_dataset, batch_size=config.DEV_BATCH_SIZE, num_workers=2)
# weight=torch.Tensor([0.06991771051950575,
#  0.8684900102905428,
#  0.3884940460347374,
#  0.1632070966173379,
#  0.3837762350309036,
#  0.17404046311724222,
#  0.5530579155791235,
#  0.7180157013921196,
#  0.4019843071440322,
#  1.0])
# weight=weight.to(device)
weight=1

model = EntityModel(num_punct=10,weight=weight)
for i,param in enumerate(model.bert.parameters()):
    if i<198-config.UNFROZEN_LAYERS:
        param.requires_grad = False
    else:
        param.requires_grad = True
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

num_train_steps = train_dataset.tensors[0].size()[0] / config.TRAIN_BATCH_SIZE * config.EPOCHS

base_opt = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
optimizer = SWA(base_opt)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps
)

best_loss = np.inf
for epoch in range(config.EPOCHS):
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
    print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
    if test_loss < best_loss:
        torch.save(model.state_dict(), config.MODEL_PATH)
        best_loss = test_loss