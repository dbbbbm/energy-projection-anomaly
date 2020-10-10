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
ITERS = 50
ALPHA = 0.1
LR = 0.1
DATA_PATH = '../mvtec_anomaly/'
CLASSES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
           'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

def visualize(opt):
    def optimize(x, model):
        x_0 = x.detach().clone()
        x.requires_grad_()
        for i in range(ITERS):
            out = model(x).detach()
            loss = torch.sum((x - out)**2) + ALPHA * torch.abs(x - x_0).sum()
            loss.backward()
            with torch.no_grad():
                x_grad = x.grad.data
                x = x - LR * x_grad * (x - out)**2
            x.requires_grad_()
        return x
    device = torch.device('cuda:{}'.format(opt.cuda))
    model = models.AE(opt.ls, opt.mp, img_size=IMG_SIZE)
    # model = models.Spatial2DAE()
    model.load_state_dict(torch.load(
        './models/%s.pth' % opt.exp, map_location='cpu'))
    model.to(device)
    model.eval()
    path = os.path.join(DATA_PATH, CLASSES[opt.c], 'test')
    mask_path = os.path.join(DATA_PATH, CLASSES[opt.c], 'ground_truth')


    y_score, y_true = [], []

    try:
        os.makedirs('./visual_grad/{}'.format(opt.exp))
    except:
        pass

    for type_ in os.listdir(path):
        for img_name in os.listdir(os.path.join(path, type_)):
            img = Image.open(os.path.join(path, type_, img_name)).resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
            img = TF.to_tensor(img).unsqueeze(0)
            if img.size(1) == 1:
                img = img.repeat(1, 3, 1, 1)
            if type_ != 'good':
                mask = Image.open(os.path.join(mask_path, type_, img_name.split('.')[0]+'_mask.png')).resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
                mask = TF.to_tensor(mask)
                mask = mask.unsqueeze(0).repeat(1, 3, 1, 1)
                y_true.append(1)
            else:
                mask = torch.zeros_like(img)
                y_true.append(0)
            img = img.to(device)

            img = (img - 0.5) / 0.5

            result = optimize(img, model)
            rec_err = (result - img) ** 2
            rec_err = rec_err.mean(dim=1, keepdim=True)

            img = torch.clamp(img.cpu() * 0.5 + 0.5, 0, 1)
            rec = torch.clamp(result.cpu() * 0.5 + 0.5, 0, 1)
            rec_err = torch.clamp(rec_err.cpu() / 0.5, 0, 1).repeat(1, 3, 1, 1)
            y_score.append(rec_err.mean().item())

            cat = torch.cat((mask, img, rec, rec_err))

            try:
                os.mkdir('./visual_grad/{}/{}'.format(opt.exp, type_))
            except:
                pass
            
            save_image(cat, './visual_grad/{}/{}/{}'.format(opt.exp, type_, img_name))

    y_score, y_true = np.array(y_score), np.array(y_true)

    print(metrics.roc_auc_score(y_true, y_score))





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', dest='cuda', type=int, default=0)
    parser.add_argument('--exp', dest='exp', type=str, default='myae256')
    parser.add_argument('--ls', dest='ls', type=int, default=16)
    parser.add_argument('--class', dest='c', type=int, default=0)
    parser.add_argument('--mp', dest='mp', type=float, default=1)
    opt = parser.parse_args()
    device = torch.device('cuda:{}'.format(opt.cuda))
    torch.cuda.set_device('cuda:{}'.format(opt.cuda))
    opt.exp += '_class={}'.format(CLASSES[opt.c])

    visualize(opt)