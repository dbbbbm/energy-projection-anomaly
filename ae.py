import models
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import mvtec_data
import torchvision.transforms.functional as TF
from ssim import SSIM
from skimage.feature import local_binary_pattern
from torchvision.utils import save_image
from PIL import Image
from sklearn import metrics, neighbors, mixture, svm
from sklearn import decomposition, manifold
from tqdm import tqdm
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from torchvision.utils import save_image

BATCH_SIZE = 64
WORKERS = 4
torch.backends.cudnn.benchmark = True
IMG_SIZE = 256
DATA_PATH = '../mvtec_anomaly_detection/'
CLASSES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
           'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
# gray-scale: 4-grid 9-screw 14-zipper


def train(opt):
    model = models.AE(opt.ls, opt.mp, img_size=IMG_SIZE)
    # model = models.Spatial2DAE()
    model.to(device)
    EPOCHS = 250
    loader = mvtec_data.get_dataloader(
        DATA_PATH, CLASSES[opt.c], 'train', BATCH_SIZE, WORKERS, IMG_SIZE)
    test_loader = mvtec_data.get_dataloader(
        DATA_PATH, CLASSES[opt.c], 'test', BATCH_SIZE, WORKERS, IMG_SIZE)

    # opt.model_dir = '/media/user/disk/models/'
    opt.model_dir = './models'
    opt.epochs = EPOCHS
    train_loop(model, loader, test_loader, opt)


def train_loop(model, loader, test_loader, opt):
    device = torch.device('cuda:{}'.format(opt.cuda))
    print(opt.exp)
    optim = torch.optim.Adam(model.parameters(), 5e-4, betas=(0.5, 0.999))
    writer = SummaryWriter('tblog/%s' % opt.exp)
    for e in tqdm(range(opt.epochs)):
        losses = []
        model.train()
        for (x, _) in tqdm(loader):
            x = x.to(device)
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)

            x.requires_grad = False
            out = model(x)
            rec_err = (out - x) ** 2
            loss = rec_err.mean()
            losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

        losses = np.mean(losses)
        writer.add_scalar('rec_err', losses, e)
        writer.add_images('recons', torch.cat((x, out)).cpu()*0.5+0.5, e)
        print('epochs:{}, recon error:{}'.format(e, losses))

    torch.save(model.state_dict(), 'models/{}.pth'.format(opt.exp))



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--u', dest='u', action='store_true')
    parser.add_argument('--gpu', dest='cuda', type=int, default=0)
    parser.add_argument('--exp', dest='exp', type=str, default='myae256')
    parser.add_argument('--ls', dest='ls', type=int, default=16)
    parser.add_argument('--class', dest='c', type=int, default=0)
    parser.add_argument('--mp', dest='mp', type=float, default=1)
    opt = parser.parse_args()
    device = torch.device('cuda:{}'.format(opt.cuda))
    torch.cuda.set_device('cuda:{}'.format(opt.cuda))
    opt.exp += '_class={}'.format(CLASSES[opt.c])
    print('start ae training...')
    train(opt)
