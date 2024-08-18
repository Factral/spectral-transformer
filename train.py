from models import Unet, ConvNext, FCN, unetsmp
from torchinfo import summary
from dataloader import FacadeDataset
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import Metrics
from utils import make_plot_val

from parser import get_arg_parser
import wandb
import torch
import torch.nn as nn


# ---------------
# training function
# ---------------
def train_one_epoch(model, data_loader, optimizer, lossfunc):
    global device, scaler, model_reconstruct, criterion_reconstruct, metric_train

    model.train()
    running_loss = []

    for cubes, rgbs, labels in tqdm(data_loader):

        rgbs = rgbs.to(device)
        labels = labels.to(device)
        cubes = cubes.to(device)

        with torch.cuda.amp.autocast(dtype=torch.float16):

            if args.usergb:
                if args.reconstruct:
                    cubes_reconstructed = model_reconstruct(rgbs)
                    outputs = model(cubes_reconstructed)
                else:
                    outputs = model(rgbs)
            else:
                outputs = model(cubes)

            if args.reconstruct and args.regularize:
                loss_seg = lossfunc(outputs, labels.long())
                loss_rec = criterion_reconstruct(cubes_reconstructed, cubes)
                wandb.log({'loss_seg_train': loss_seg, 'loss_rec_train': loss_rec})
                loss = loss_seg + loss_rec
            else:
                loss = lossfunc(outputs, labels.long())

        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss.append(loss.item())

        pred = nn.functional.softmax(outputs, dim=1)
        metric_train.update(pred.argmax(1), labels.long())


    pixel_acc, macc, miou = metric_train.compute()
    metric_train.reset()

    epoch_loss = sum(running_loss) / len(data_loader.dataset)

    return epoch_loss, pixel_acc, macc, miou


# ---------------
# validation function
# ---------------
def validate(model, data_loader, lossfunc):
    global metric_val, device, args, scheduler

    model.eval()
    running_loss = []

    with torch.inference_mode():
        for cubes, rgbs,  labels in tqdm(data_loader):
            rgbs = rgbs.to(device)
            labels = labels.to(device)
            cubes = cubes.to(device)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                if args.usergb:
                    if args.reconstruct:
                        cubes_reconstructed = model_reconstruct(rgbs)
                        outputs = model(cubes_reconstructed)
                    else:
                        outputs = model(rgbs)
                else:
                    outputs = model(cubes)

                if args.reconstruct and args.regularize:
                    loss_seg = lossfunc(outputs, labels.long())
                    loss_rec = criterion_reconstruct(cubes_reconstructed, cubes)
                    wandb.log({'loss_seg_val': loss_seg, 'loss_rec_val': loss_rec})
                else:
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


def main(args):
    global device, scaler, model_reconstruct, criterion_reconstruct, metric_val, metric_train, scheduler

    if args.wandb:
        wandb.login(key='fe0119224af6709c85541483adf824cec731879e')
        wandb.init(project="transformer-material-segmentation", name=args.exp_name, group=args.group)
        wandb.config.update(args)


    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()


    # ---------------
    # dataloaders
    # ---------------
    dataset_train = FacadeDataset(Path(args.datadir) / 'train', repeatrgb=args.repeatrgb)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8,
    pin_memory=True, drop_last=True)

    dataset_test = FacadeDataset(Path(args.datadir) / 'test', repeatrgb=args.repeatrgb)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=8,
    pin_memory=True)


    # ---------------
    # model selection
    # ---------------
    if args.usergb:
        if args.reconstruct:
            n_channels = 31
        else:
            n_channels = 3
    else:
        n_channels = 31
    n_classes  = 44


    if args.model == 'unet':
        model = Unet(n_channels, n_classes)
    if args.model == 'convnext':
        model = ConvNext(n_channels, n_classes)
    if args.model == 'fcn':
        model = FCN(n_channels, n_classes)
    if args.model == 'unetsmp':
        model = unetsmp(n_channels, n_classes)


    if args.reconstruct:
        model_reconstruct = Unet(3, 31)
        model_reconstruct = model_reconstruct.to(device)

    model = model.to(device)

    if args.weights:
        model.load_weights(args.weights)


    # ---------------
    # training params
    # ---------------
    criterion = torch.nn.CrossEntropyLoss().to(device)
    if args.reconstruct:
        criterion_reconstruct = torch.nn.MSELoss().to(device)


    if not args.reconstruct:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(model_reconstruct.parameters()), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15, factor=0.5, verbose=True,  min_lr=1e-5)
    metric_val = Metrics()
    metric_train = Metrics()


    summary(model, input_size=(args.batch_size, n_channels, 512, 512))
    if args.reconstruct:
        print('\nReconstruct model\n')
        summary(model_reconstruct, input_size=(args.batch_size, 3, 512, 512))


    # ---------------
    # training loop
    # ---------------
    best_val_miou = 0
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}\n-------------------------------')

        #train one epoch
        epoch_loss, train_pixel_acc, train_macc, train_miou  = train_one_epoch(model, dataloader_train, optimizer, criterion)

        #validate
        val_loss, fig_val, pixel_acc, macc, miou = validate(model, dataloader_test, criterion)

        #save checkpoint
        if miou > best_val_miou:
            best_val_miou = miou
            torch.save(model.state_dict(), f'checkpoints/{args.exp_name}_best_model.pth')
            print('Model outperformed previous best. Saving model... ')

        if args.wandb:
            wandb.log({
                    'train_loss': epoch_loss,
                    'val_loss': val_loss,
                    'val_fig': fig_val,
                    'epoch': epoch,
                    'pixel_acc_val': pixel_acc, 'test_macc_val': macc, 'miou_val': miou,
                    'pixel_acc_train': train_pixel_acc, 'macc_train': train_macc, 'miou_train': train_miou,
                    'lr': optimizer.param_groups[0]['lr']
                    })

        print(f'Epoch {epoch} train loss: {epoch_loss:.4f}, val loss: {val_loss:.4f}')
        print(f'Pixel Acc: {pixel_acc:.4f}, mAcc: {macc:.4f}, mIoU: {miou:.4f}')

    print("finished training")
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)