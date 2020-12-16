import numpy as np
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch
from config import *
from engine import *
from model import EntityModel


#torch.set_default_tensor_type(torch.cuda.FloatTensor)
#device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
train_dataset=torch.load('../../data/ted-train.pt')
dev_dataset=torch.load('../../data/ted-dev.pt')
train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)
dev_dataloader=torch.utils.data.DataLoader(dev_dataset, batch_size=config.DEV_BATCH_SIZE, num_workers=2)


model = EntityModel(num_punct=10)
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

optimizer = AdamW(optimizer_parameters, lr=3e-5)
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
    test_loss = eval_fn(
        valid_dataloader,
        model,
        device
    )
    print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
    if test_loss < best_loss:
        torch.save(model.state_dict(), config.MODEL_PATH)
        best_loss = test_loss
