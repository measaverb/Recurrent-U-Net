import argparse

import torch
from torch.utils.data import DataLoader

from RecurrentUNet.dataset import Vein
from RecurrentUNet.transform import preprocessing
from RecurrentUNet.models.RUNet import R_UNet
from RecurrentUNet.loss import dice_loss
from RecurrentUNet.metric import dice_coeff
from RecurrentUNet.trainer import Trainer

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch_size', type=int, default=1
)
parser.add_argument(
    '--epoch', type=int, default=40
)
parser.add_argument(
    '--lr', type=float, default=0.001
)
parser.add_argument(
    '--dataset', type=str, default='./data/'
)
parser.add_argument(
    '--workers', type=int, default=4
)
parser.add_argument(
    '--save_model', type=str, default='./save_model/'
)

cfg = parser.parse_args()
print(cfg)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == "__main__":
    ds_train = Vein(root='./data/', split='train', transform=preprocessing)
    ds_test = Vein(root='./data/', split='test', transform=preprocessing)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)

    print("DATA LOADED")
    model = R_UNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = dice_loss
    success_metric = dice_coeff
    summary = SummaryWriter()

    trainer = Trainer(model, criterion, optimizer, success_metric, device, summary, False)
    fit = trainer.fit(dl_train, dl_test, num_epochs=cfg.epoch, checkpoints=cfg.save_model+model.__class__.__name__+'.pt')
    torch.save(model.state_dict(), './unet/final_state_dict.pt')
    torch.save(model, './unet/final.pt')

    loss_fn_name = "cross entropy"
    best_score = str(fit.best_score)
    print(f"Best loss score(loss function = {loss_fn_name}): {best_score}")

