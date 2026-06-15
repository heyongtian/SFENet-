import csv

from PIL.Image import Image

from dataset import MyDataset
from model.SFENet import SFENet
import torch
import argparse
from torch.utils.data import DataLoader
from utils import losses
import os

from data.data_provider import SingleLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import numpy as np
# import model
from utils.metric import calculate_psnr, calculate_ssim, calculate_snr
from utils.training_util import save_checkpoint,MovingAverage, load_checkpoint
# from collections import OrderedDict
import torch.nn as nn
torch.backends.cudnn.enabled = False

def train(args):
    torch.set_num_threads(args.num_workers)
    torch.manual_seed(0)

    full_dataset = MyDataset(args.noise_dir, args.gt_dir)
    valida_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - valida_size * 2
    # 划分数据集
    train_dataset, test_dataset, valida_dataset = torch.utils.data.random_split(full_dataset,
                                                                                [train_size, valida_size, valida_size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = nn.L1Loss()
    checkpoint_dir = args.checkpoint
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model = SFENet().to(device)

    log_file = os.path.join(checkpoint_dir, "validation_log.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Validation Loss", "Validation PSNR"])  # 写入表头
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    optimizer.zero_grad()
    average_loss = MovingAverage(args.save_every)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=args.lr)

    if args.restart:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        print('=> no checkpoint file to be loaded.')
    else:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', args.load_type)
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            state_dict = checkpoint['state_dict']

            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            print('=> no checkpoint file to be loaded.')
    for epoch in range(start_epoch, args.epoch):
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        valida_loader = torch.utils.data.DataLoader(dataset=valida_dataset, batch_size=args.batch_size, shuffle=True)
        psnrs = []
        total_loss = 0.0
        model.train()
        for step, (noise, gt) in enumerate(train_loader):
            noise = noise.to(device=device, dtype=torch.float32)
            gt = gt.to(device=device, dtype=torch.float32)
            pred = model(noise)
            loss = loss_func(pred, gt)
            total_loss += loss.item()
            psnr_x_ = calculate_psnr(pred, gt)
            psnrs.append(psnr_x_)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_loss.update(loss)
            if global_step % args.save_every == 0:
                if average_loss.get_value() < best_loss:
                    is_best = True
                    best_loss = average_loss.get_value()
                else:
                    is_best = False

                save_dict = {
                    'epoch': epoch,
                    'global_iter': global_step,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(save_dict, is_best, checkpoint_dir, global_step)
            global_step += 1
        psnr_avg = np.mean(psnrs)
        total_loss /= len(train_loader)
        print("epoch={}，训练集PSNR：{:.4f}，Loss：{:.4f}".format(epoch, psnr_avg, total_loss))
        model.eval()
        validation_loss = 0
        psnrs = []
        ssims = []
        snrs= []
        with torch.no_grad():
            for val_noise, val_gt in valida_loader:
                val_noise = val_noise.to(device=device, dtype=torch.float32)
                val_gt = val_gt.to(device=device, dtype=torch.float32)
                val_pred = model(val_noise)
                validation_loss += loss_func(val_pred, val_gt).item()
                snr_x = calculate_snr(val_pred, val_gt)
                psnr_x = calculate_psnr(val_pred, val_gt)
                ssim_x = calculate_ssim(val_pred, val_gt)
                snrs.append(snr_x)
                psnrs.append(psnr_x)
                ssims.append(ssim_x)
        validation_loss /= len(valida_loader)
        avg_snr = np.mean(snrs)
        avg_psnr = np.mean(psnrs)
        avg_ssim = np.mean(ssims)
        print(f"Validation Loss: {validation_loss:.4f}, SNR: {avg_snr:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, validation_loss, avg_psnr])

        print('Epoch {} is finished.'.format(epoch))
        scheduler.step()



if __name__ == "__main__":
    argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir', '-n', default='./data/sgy_data_real/feature', help='path to noise folder image')
    parser.add_argument('--gt_dir', '-g', default='./data/sgy_data_real/label', help='path to gt folder image')
    parser.add_argument('--image_size', '-sz', default=64, type=int, help='size of image')

    parser.add_argument('--batch_size', '-bs', default=28, type=int, help='batch size')
    parser.add_argument('--epoch', '-e', default=50, type=int, help='batch size')
    parser.add_argument('--save_every', '-se', default=1, type=int, help='save_every')
    parser.add_argument('--loss_every', '-le', default=1, type=int, help='loss_every')
    # parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--restart', '-r', action='store_true',
                        help='Whether to remove all old files and restart the training process')
    parser.add_argument('--num_workers', '-nw', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoints',
                        help='the checkpoint to eval')
    parser.add_argument('--load_type', "-l" ,default="best", type=str, help='Load type best_or_latest ')


    args = parser.parse_args()
    #
    train(args)