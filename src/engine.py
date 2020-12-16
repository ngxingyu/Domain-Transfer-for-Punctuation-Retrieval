import torch
import model
from tqdm import tqdm
def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for v in data:
            v = v.to(device,non_blocking=True)
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
        for v in data:
            v = v.to(device)
        _, loss = model(data)
        final_loss += loss.item()
    return final_loss / len(data_loader)
