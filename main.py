# *coding:utf-8 *


import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import warnings
warnings.filterwarnings('ignore')

from time import time
from Visualizer import Visualizer
from networks.cenet import CE_Net_
from framework import MyFrame
from loss import dice_bce_loss
from PIL import Image
from data import ImageFolder
import cv2
import numpy as np
from Metrics import calculate_auc_test, accuracy


def train_CE_Net_Vessel():
    IMAGE_SHAPE = (448, 448)
    Use_Test = False
    ROOT = '/data/zaiwang/Dataset/ORIGA'
    NAME = 'Unet-origin-' + ROOT.split('/')[-1]
    BATCH_SIZE_PER_CARD = 12
    viz = Visualizer(env=NAME)

    solver = MyFrame(CE_Net_, dice_bce_loss, 2e-4)

    batch_size = torch.cuda.device_count() * BATCH_SIZE_PER_CARD

    # Preparing the dataloader

    dataset = ImageFolder(root_path=ROOT, datasets='ORIGA')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    mylog = open('logs/' + NAME + '.log', 'w')

    tic = time()
    no_optim = 0
    total_epoch = 500
    train_epoch_best_loss = 10000.

    for epoch in range(0, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0

        index = 0

        for img, mask in data_loader_iter:
            solver.set_input(img, mask)

            train_loss, pred = solver.optimize()

            train_epoch_loss += train_loss

            index = index + 1

            if index % 10 == 0:
                # train_epoch_loss /= index
                # viz.plot(name='loss', y=train_epoch_loss)
                show_image = (img + 1.6) / 3.2 * 255.
                viz.img(name='images', img_=show_image[0, :, :, :])
                viz.img(name='labels', img_=mask[0, :, :, :])
                viz.img(name='prediction', img_=pred[0, :, :, :])

        show_image = (img + 1.6) / 3.2 * 255.
        viz.img(name='images', img_=show_image[0, :, :, :])
        viz.img(name='labels', img_=mask[0, :, :, :])
        viz.img(name='prediction', img_=pred[0, :, :, :])

        train_epoch_loss = train_epoch_loss / len(data_loader_iter)
        print(mylog, '********')
        print(mylog, 'epoch:', epoch, '    time:', int(time() - tic))
        print(mylog, 'train_loss:', train_epoch_loss)
        print(mylog, 'SHAPE:', IMAGE_SHAPE)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        print('SHAPE:', IMAGE_SHAPE)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('./weights/' + NAME + '.th')
        if no_optim > 20:
            print(mylog, 'early stop at %d epoch' % epoch)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > 10:
            if solver.old_lr < 5e-7:
                break
            # solver.load('./weights/' + NAME + '.th')
            solver.update_lr(2.0, factor=True, mylog=mylog)
            no_optim = 0
        mylog.flush()

    print(mylog, 'Finish!')
    print('Finish!')
    mylog.close()

if __name__ == '__main__':
    print(torch.cuda.device_count())
    train_CE_Net_Vessel()