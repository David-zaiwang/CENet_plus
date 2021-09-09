import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import sklearn.metrics as metrics
import cv2
import os
import numpy as np

from time import time
from PIL import Image
from Metrics import dice, mask_iou, mask_to_boundary

import warnings

warnings.filterwarnings('ignore')

from networks.cenet import CE_Net_

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

BATCHSIZE_PER_CARD = 8


def calculate_auc_test(prediction, label):
    # read images
    # convert 2D array into 1D array
    result_1D = prediction.flatten()
    label_1D = label.flatten()

    label_1D = label_1D / 255

    auc = metrics.roc_auc_score(label_1D.astype(np.uint8), result_1D)

    # print("AUC={0:.4f}".format(auc))

    return auc


def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    sen = TP / (TP + FN)
    return acc, sen


class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True, without_TTA=False):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if without_TTA:
            return self.test_one_img_without_test_aug(path)
        elif batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_without_test_aug(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (448, 448))
        img = np.expand_dims(img, 0)
        img = img.transpose(0, 3, 1, 2)
        img = V(torch.Tensor(np.array(img, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        mask = self.net.forward(img).squeeze().cpu().data.numpy()
        return mask

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img = cv2.resize(img, (448, 448))

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def load(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model)


def dice_coefficient(a, b):
    a_bigrams = a.astype(np.uint8)
    b_bigrams = b.astype(np.uint8)
    overlap = a_bigrams + b_bigrams
    overlap = np.sum(overlap == 2)
    return overlap * 2.0 / (np.sum(a_bigrams) + np.sum(b_bigrams))


def test_ce_net_ORIGA(root_path, weight_path):
    # root_path = '/data/zaiwang/Dataset/ORIGA_OD'
    without_TTA = True
    test_dataset_category_name = root_path.split('/')[-1]
    if test_dataset_category_name == 'ORIGA':
        read_files = os.path.join(root_path, 'Set_B.txt')

        image_root = os.path.join(root_path, 'images')
        gt_root = os.path.join(root_path, 'masks')

        images_list = []
        masks_list = []
        for image_name in open(read_files):
            image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
            label_path = os.path.join(gt_root, image_name.split('.')[0] + '.jpg')



            images_list.append(image_path)
            masks_list.append(label_path)
    elif test_dataset_category_name == 'Messidor':
        read_files = os.path.join(root_path, 'test.txt')

        image_root = os.path.join(root_path, 'save_image')
        gt_root = os.path.join(root_path, 'save_mask')

        images_list = []
        masks_list = []
        for image_name in open(read_files):
            image_path = os.path.join(image_root, image_name.split('.')[0] + '.png')
            label_path = os.path.join(gt_root, image_name.split('.')[0] + '.png')

            images_list.append(image_path)
            masks_list.append(label_path)
    elif test_dataset_category_name == 'ORIGA_OD':
        read_files = os.path.join(root_path, 'Set_B.txt')

        image_root = os.path.join(root_path, 'crop_image')
        gt_root = os.path.join(root_path, 'crop_mask')

        images_list = []
        masks_list = []
        for image_name in open(read_files):
            image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
            label_path = os.path.join(gt_root, image_name.split('.')[0] + '.png')

            images_list.append(image_path)
            masks_list.append(label_path)
    elif test_dataset_category_name == 'humanseg':
        image_root = '/data/zaiwang/Dataset/humanseg/val/imgs'
        gt_root = '/data/zaiwang/Dataset/humanseg/val/masks'

        images_list = []
        masks_list = []
        for image_name in os.listdir(image_root):
            image_path = os.path.join(image_root, image_name)
            label_path = os.path.join(gt_root, image_name)

            if cv2.imread(image_path) is not None:

                if os.path.exists(image_path) and os.path.exists(label_path):
                    images_list.append(image_path)
                    masks_list.append(label_path)

    else:
        images_list = []
        masks_list = []
    solver = TTAFrame(CE_Net_)
    # solver.load('weights/log01_dink34-DCENET-DRIVE.th')
    # solver.load('./weights/boundary_iou-ORIGA-v1.th')
    # weight_path = './weights/CE_Net_boundary_dice_bce_loss-Messidor-v1.th'
    # weight_path = './weights/CE_Net_boundary_dice_bce_loss-ORIGA_OD-v2.th'
    # weight_path = './weights/CE_Net_boundary_dice_bce_loss-ORIGA_OD-v3.th'
    # weight_path = './weights/CE_Net_boundary_dice_bce_loss-ORIGA_OD-v4.th'
    solver.load(weight_path)
    tic = time()
    NAME = weight_path.split('/')[-1].replace('.th', '')
    target = './submits/' + NAME + '/'
    if not os.path.exists(target):
        os.mkdir(target)
    total_m1 = 0

    hausdorff = 0
    total_acc = []
    total_sen = []
    total_dice_error = []
    total_mask_iou = []
    total_boundary_mask_iou = []
    if without_TTA:
        threshold = 0.5
    else:
        threshold = 4
    total_auc = []

    disc = 20
    if len(os.listdir(target)) == len(masks_list) + 1:
        print("This model has been evaluated, and the results could directly be used to ")
        for i in range(len(masks_list)):
            image_path = images_list[i]
            ground_truth_path = masks_list[i]

            ground_truth = np.array(Image.open(ground_truth_path))
            mask_path = target + image_path.split('/')[-1].split('.')[0] + '-mask.png'
            mask = cv2.imread(mask_path)

            new_mask = mask[:, :, 0].copy()
            total_auc.append(calculate_auc_test(new_mask / 8., ground_truth))

            predi_mask = np.zeros(shape=np.shape(mask))
            predi_mask[mask > disc] = 1
            gt = np.zeros(shape=np.shape(ground_truth))
            gt[ground_truth > 0] = 1

            acc, sen = accuracy(predi_mask[:, :, 0], gt)
            # dice = dice_coefficient(predi_mask[:, :, 0], gt)
            dice_coe = dice(predi_mask[:, :, 0], gt)
            dice_error = 1 - dice_coe

            iou_error = 1 - mask_iou(predi_mask[:, :, 0], gt)
            total_mask_iou.append(iou_error)

            # boundary mask IOU
            prediction_boundary_mask = mask_to_boundary(predi_mask[:, :, 0])
            gt_boundary_mask = mask_to_boundary(gt)
            boundary_mask_iou = 1 - mask_iou(prediction_boundary_mask, gt_boundary_mask)
            total_boundary_mask_iou.append(boundary_mask_iou)

            total_dice_error.append(dice_error)
            total_acc.append(acc)
            total_sen.append(sen)

            print(i + 1, acc, sen, calculate_auc_test(new_mask / 8., ground_truth), dice_error, iou_error,
                  boundary_mask_iou)
    else:
        for i in range(len(masks_list)):
            image_path = images_list[i]

            mask = solver.test_one_img_from_path(image_path, without_TTA=without_TTA)

            new_mask = mask.copy()

            mask[mask > threshold] = 255
            mask[mask <= threshold] = 0
            mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)

            ground_truth_path = masks_list[i]

            # print(ground_truth_path)
            ground_truth = np.array(Image.open(ground_truth_path))

            mask = cv2.resize(mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0]))

            new_mask = cv2.resize(new_mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0]))
            total_auc.append(calculate_auc_test(new_mask / 8., ground_truth))

            predi_mask = np.zeros(shape=np.shape(mask))
            predi_mask[mask > disc] = 1
            gt = np.zeros(shape=np.shape(ground_truth))
            gt[ground_truth > 0] = 1

            acc, sen = accuracy(predi_mask[:, :, 0], gt)
            # dice = dice_coefficient(predi_mask[:, :, 0], gt)
            dice_coe = dice(predi_mask[:, :, 0], gt)
            dice_error = 1 - dice_coe

            iou_error = 1 - mask_iou(predi_mask[:, :, 0], gt)
            total_mask_iou.append(iou_error)

            # boundary mask IOU
            prediction_boundary_mask = mask_to_boundary(predi_mask[:, :, 0])
            gt_boundary_mask = mask_to_boundary(gt)
            boundary_mask_iou = 1 - mask_iou(prediction_boundary_mask, gt_boundary_mask)
            total_boundary_mask_iou.append(boundary_mask_iou)

            total_dice_error.append(dice_error)
            total_acc.append(acc)
            total_sen.append(sen)

            print(i + 1, acc, sen, calculate_auc_test(new_mask / (2 * threshold), ground_truth), dice_error, iou_error,
                  boundary_mask_iou)
            name = image_path.split('/')[-1]
            cv2.imwrite(target + name.split('.')[0] + '-mask.png', mask.astype(np.uint8))

    print("-------------------------------------------")
    print('ID, ACC, Sen, AUC, DICE_error, MASK_error, boundary_error')
    print("total_acc mean : {0:.5f}, and std : {0:.5f}".format(np.mean(total_acc), np.std(total_acc)))
    print("total_sen mean : {0:.5f}, and std : {0:.5f}".format(np.mean(total_sen), np.std(total_sen)))
    print("total_auc mean : {0:.5f}, and std : {0:.5f}".format(np.mean(total_auc), np.std(total_auc)))
    print("total_dice error mean : {0:.5f}, and std : {0:.5f}".format(np.mean(total_dice_error),
                                                                      np.std(total_dice_error)))
    print("total_mask error mean : {0:.5f}, and std : {0:.5f}".format(np.mean(total_mask_iou), np.std(total_mask_iou)))
    print("total_bound error mean : {0:.5f}, and std : {0:.5f}".format(np.mean(total_boundary_mask_iou),
                                                                       np.std(total_boundary_mask_iou)))


if __name__ == '__main__':
    root_path = '/data/zaiwang/Dataset/humanseg'
    weight_path = './weights/CE_Net_boundary_dice_bce_loss-humanseg-v1.th'
    test_ce_net_ORIGA(root_path, weight_path)
