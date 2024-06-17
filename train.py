from models import Unet
from torchinfo import summary
from dataloader import FacadeDataset
from pathlib import Path
from torch.utils.data import DataLoader

import argparse
import wandb
import torch


parser = argparse.ArgumentParser(description='train transformer por material segmentation.')
parser.add_argument('--exp_name', type=str, help='Experiment name', required=True)
parser.add_argument('--model', type=str, help='Model to train', required=True)
parser.add_argument('--datadir', type=str, help='Path to the dataset', required=True)
parser.add_argument('--gpu', type=int, help='GPU number', required=True)
parser.add_argument('--epochs', type=int, help='Number of epochs', required=True)
parser.add_argument('--batch_size', type=int, help='Batch size', required=True)
parser.add_argument('--lr', type=float, help='Learning rate', required=True)
args = parser.parse_args()


wandb.login(key='fe0119224af6709c85541483adf824cec731879e')
wandb.init(project="transformer-material-segmentation", name=args.exp_name)
wandb.config.update(args)


device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')


# ---------------
# dataloaders
# ---------------
dataset_train = FacadeDataset(Path(args.datadir) / 'train')
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=-1,
 pin_memory=True)

dataset_test = FacadeDataset(Path(args.datadir) / 'test')
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=-1,
 pin_memory=True)


# ---------------
# model selection
# ---------------
if args.model == 'unet':
    model = Unet(3, 10)

# ---------------
# architecture params
# ---------------
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)




summary(model, input_size=(args.batch_size, 3, 512, 512))

