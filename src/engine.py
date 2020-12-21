import torch
from .model import *
from tqdm import tqdm
def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        if torch.cuda.is_available(): data=[_data.to(device) for _data in data]
        optimizer.zero_grad()
        _, loss = model(data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        if torch.cuda.is_available(): data=[_data.to(device) for _data in data]
        punct, loss = model(data)
        final_loss += loss.item()
    return final_loss / len(data_loader)
