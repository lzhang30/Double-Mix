import os
import math
from tqdm import tqdm
import numpy as np
import random
import SimpleITK as sitk

import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.config import Config




def EMA(cur_weight, past_weight, momentum=0.9):
    new_weight = momentum * past_weight + (1 - momentum) * cur_weight
    return new_weight


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model



def print_func(item):
    # print(type(item))
    if type(item) == torch.Tensor:
        return [round(x,4) for x in item.data.cpu().numpy().tolist()]
    elif type(item) == np.ndarray:
        return [round(x,4) for x in item.tolist()]
    else:
        raise TypeError


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / (x_sum)
    return s

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

def maybe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    itk_arr = sitk.GetArrayFromImage(itk_img)
    return itk_arr


def read_list(split, task="synapse"):

    config = Config(task)
    
    ids_list = np.loadtxt(
        os.path.join(config.save_dir, 'splits', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)


def read_data(data_id, task, nifti=False, test=False, normalize=False,unlabeled = False):
    config = Config(task)
    im_path = os.path.join(config.save_dir, 'npy', f'{data_id}_image.npy')
    if not unlabeled:
    
        lb_path = os.path.join(config.save_dir, 'npy', f'{data_id}_label.npy')
        if not os.path.exists(im_path) or not os.path.exists(lb_path):
            print((f'data_id: {data_id}'))
            raise ValueError(data_id)
        
        image = np.load(im_path)
        label = np.load(lb_path)
    else:
        image = np.load(im_path)
        label = np.zeros_like(image)
    
    if normalize:
        if task == 'chd':
            min_val = np.percentile(image, 5)
            max_val = np.percentile(image, 95)
            image = image.clip(min=min_val, max=max_val)
        elif task == 'colon':
            image = image.clip(min=-250, max=275)
        else:
            image = image.clip(min=-75, max=275)
        image = (image - image.min()) / (image.max() - image.min())
        image = image.astype(np.float32)
    
    return image, label


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fetch_data(batch, labeled=True):
    image = batch['image'].cuda()
    if labeled:
        label = batch['label'].cuda().unsqueeze(1)
        return image, label
    else:
        return image


def test_all_case(net, ids_list, task, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, _ = read_data(data_id, task, test=True, normalize=True)
        pred, _ = test_single_case(
            net, 
            image, 
            stride_xy,
            stride_z,
            patch_size,
            num_classes=num_classes
        )
        out = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes):
    image = image[np.newaxis]
    _, dd, ww, hh = image.shape
    # print(image.shape)
    # resize_shape=(patch_size[0]+patch_size[0]//4,
    #               patch_size[1]+patch_size[1]//4,
    #               patch_size[2]+patch_size[2]//4)
    #
    # image = torch.FloatTensor(image).unsqueeze(0)
    # image = F.interpolate(image, size=resize_shape,mode='trilinear', align_corners=False)
    # image = image.squeeze(0).numpy()

    image = image.transpose(0, 3, 2, 1) # <-- take care the shape
    # print(image.shape)
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    
    score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)
    # print("score_map", score_map.shape)
    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # print("test", test_patch.shape)
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # print("===",test_patch.size())
                # <-- [1, 1, Z, Y, X] => [1, 1, X, Y, Z]
                test_patch = test_patch.transpose(2, 4)
                y1 = net(test_patch) # <--
                y = F.softmax(y1, dim=1) # <--
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    # print("score_map", score_map.shape)
    # print("score_map", cnt.shape)

    score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
    score_map = score_map.transpose(0, 3, 2, 1) # => [X, Y, Z]
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map



def test_all_case_AB(net_A, net_B, ids_list, task, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, _ = read_data(data_id, task, test=True, normalize=True)
        pred, _ = test_single_case_AB(
            net_A, net_B,
            image,
            stride_xy,
            stride_z,
            patch_size,
            num_classes=num_classes
        )
        out = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')


def test_single_case_AB(net_A, net_B, image, stride_xy, stride_z, patch_size, num_classes):
    image = image[np.newaxis]
    
    _, dd, ww, hh = image.shape
    #print(image.shape)
    # resize_shape=(patch_size[0]+patch_size[0]//4,
    #               patch_size[1]+patch_size[1]//4,
    #               patch_size[2]+patch_size[2]//4)

    # image = torch.FloatTensor(image).unsqueeze(0)
    # image = F.interpolate(image, size=resize_shape,mode='trilinear', align_corners=False)
    # image = image.squeeze(0).numpy()
    
    image = image.transpose(0, 3, 2, 1) # <-- take care the shape
    # print(image.shape)
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)
    # print("score_map", score_map.shape)
    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # print("test", test_patch.shape)
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # print("===",test_patch.size())
                # <-- [1, 1, Z, Y, X] => [1, 1, X, Y, Z]
                test_patch = test_patch.transpose(2, 4)
                y1 = (net_A(test_patch) + net_B(test_patch)) / 2.0 # <--
                y = F.softmax(y1, dim=1) # <--
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    # print("score_map", score_map.shape)
    # print("score_map", cnt.shape)

    score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
    score_map = score_map.transpose(0, 3, 2, 1) # => [X, Y, Z]
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map


def eval_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, use_softmax = True):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0],
                             ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = net(test_patch)
                    # ensemble
                    if use_softmax:
                        y = torch.softmax(y1, dim=1)
                    else:
                        y = y1
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w,
                    hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad +
                                        w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map





def eval_single_case_A(netA, image, stride_xy, stride_z, patch_size, num_classes=1, use_softmax = True):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0],
                             ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = netA(test_patch)
                    # ensemble
                    if use_softmax:
                        y = torch.softmax(y1, dim=1)
                    else:
                        y = y1
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w,
                    hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad +
                                        w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map










def eval_single_case_AB(netA,netB, image, stride_xy, stride_z, patch_size, num_classes=1, use_softmax = True):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0],
                             ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = 0.5*(netA(test_patch)+netB(test_patch))
                    # ensemble
                    if use_softmax:
                        y = torch.softmax(y1, dim=1)
                    else:
                        y = y1
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w,
                    hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad +
                                        w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map


def dice_coefficient(prediction, target, num_classes):
    dice = torch.zeros(num_classes)
    epsilon = 1e-6  # 避免除以零

    # 对每个类别计算 Dice 系数
    for i in range(num_classes):
        pred_i = (prediction == i)  # 预测为类别 i 的体素
        target_i = (target == i)    # 真实标签为类别 i 的体素
        inter = (pred_i & target_i).float().sum()  # 交集
        union = pred_i.float().sum() + target_i.float().sum()  # 并集

        dice[i] = (2 * inter + epsilon) / (union + epsilon)

    return dice



