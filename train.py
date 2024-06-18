from models import Unet
from torchinfo import summary
from dataloader import FacadeDataset
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import Metrics
from utils import make_plot_val

import argparse
import wandb
import torch
import torch.nn as nn


parser = argparse.ArgumentParser(description='train transformer por material segmentation.')
parser.add_argument('--exp_name', type=str, help='Experiment name', required=True)
parser.add_argument('--model', type=str, help='Model to train', required=True)
parser.add_argument('--datadir', type=str, help='Path to the dataset', required=True)
parser.add_argument('--gpu', type=int, help='GPU number', required=True)
parser.add_argument('--epochs', type=int, help='Number of epochs', required=True)
parser.add_argument('--batch_size', type=int, help='Batch size', required=True)
parser.add_argument('--lr', type=float, help='Learning r    ate', required=True)
args = parser.parse_args()


wandb.login(key='fe0119224af6709c85541483adf824cec731879e')
wandb.init(project="transformer-material-segmentation", name=args.exp_name)
wandb.config.update(args)


device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    scaler = torch.cuda.amp.GradScaler()


# ---------------
# dataloaders
# ---------------
dataset_train = FacadeDataset(Path(args.datadir) / 'train')
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4,
 pin_memory=True)

dataset_test = FacadeDataset(Path(args.datadir) / 'test')
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4,
 pin_memory=True)


# ---------------
# model selection
# ---------------
if args.model == 'unet':
    model = Unet(31, 44)


# ---------------
# training params
# ---------------
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15, factor=0.5, verbose=True,  min_lr=1e-5)
metric_val = Metrics()


summary(model, input_size=(args.batch_size, 31, 512, 512))


# ---------------
# training function
# ---------------
def train(model, data_loader, optimizer, lossfunc):
    model.train()
    running_loss = []

    for cubes, rgbs, labels in tqdm(data_loader):

        rgbs = rgbs.to(device)
        labels = labels.to(device)
        cubes = cubes.to(device)

        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(cubes)
            loss = lossfunc(outputs, labels.long())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss.append(loss.item())

    epoch_loss = sum(running_loss) / len(data_loader.dataset)

    return epoch_loss


# ---------------
# validation function
# ---------------
def validate(model, data_loader, lossfunc):
    model.eval()
    running_loss = []

    with torch.no_grad():
        for cubes, rgbs,  labels in tqdm(data_loader):
            rgbs = rgbs.to(device)
            labels = labels.to(device)
            cubes = cubes.to(device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(cubes)
                
                loss = lossfunc(outputs, labels.long())

            running_loss.append(loss.item())

            pred = nn.functional.softmax(outputs, dim=1)
            metric_val.update(pred.argmax(1), labels.long())

    pixel_acc, macc, miou = metric_val.compute()
    metric_val.reset()
    
    val_loss = sum(running_loss) / len(running_loss)
    scheduler.step(miou)

    fig = make_plot_val(rgbs, outputs, labels)

    return val_loss, fig, pixel_acc, macc, miou


# ---------------
# training loop
# ---------------
best_val_miou = 0
for epoch in range(args.epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')

    #train one epoch
    epoch_loss  = train(model, dataloader_train, optimizer, criterion)

    #validate
    val_loss, fig_val, pixel_acc, macc, miou = validate(model, dataloader_test, criterion)

    #save checkpoint
    if best_val_miou > miou:
        best_val_miou = miou
        torch.save(model.state_dict(), 'models/best_model.pth')

    #logs
    wandb.log({
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'val_fig': fig_val,
            'epoch': epoch,
            'pixel_acc_val': pixel_acc, 'macc_val': macc, 'miou_val': miou,
            'lr': optimizer.param_groups[0]['lr']
            })

    print(f'Epoch {epoch} train loss: {epoch_loss:.4f}, val loss: {val_loss:.4f}')