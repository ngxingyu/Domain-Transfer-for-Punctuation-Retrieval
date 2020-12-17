import torch
import model
from tqdm import tqdm
def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
<<<<<<< HEAD
        if torch.cuda.is_available(): data=[_data.to(device) for _data in data]
        #for _data in data:
        #    print(_data.is_cuda)
=======
        for v in data:
            v = v.to(device,non_blocking=True)
>>>>>>> 04898a3974601c54d3c1d7e3ea11cb1461cace21
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
<<<<<<< HEAD
        if torch.cuda.is_available(): data=[_data.to(device) for _data in data]
=======
        for v in data:
            v = v.to(device)
>>>>>>> 04898a3974601c54d3c1d7e3ea11cb1461cace21
        _, loss = model(data)
        final_loss += loss.item()
    return final_loss / len(data_loader)
