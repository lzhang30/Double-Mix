import os
import sys
import logging
from tqdm import tqdm
import argparse
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_20p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_80p')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True) # <--
parser.add_argument('-ep', '--max_epoch', type=int, default=500)
parser.add_argument('--cps_loss', type=str, default='wce')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=1)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=True) # <--
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
parser.add_argument('-sm', '--start_mix', type=int, default=60)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from models.vnet import VNet
from utils import EMA, maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr, print_func, kaiming_normal_init_weight
from utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from data.data_loaders import Synapse_AMOS
from utils.config import Config
import random
config = Config(args.task)



def create_ema_model(config,model):
    ema_model = VNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def mix(mask, data=None, target=None):
    # Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] // 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(data.shape[0] // 2)]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(data.shape[0] // 2)])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target


def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N



def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    if args.cps_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.consistency_rampup is None:
            args.consistency_rampup = args.max_epoch
        return args.cps_w * sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.cps_w



def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    else:
        raise ValueError(name)


def make_loader(split, dst_cls=Synapse_AMOS, repeat=None, is_training=True, unlabeled=False):
    if is_training:
        dst = dst_cls(
            task=args.task,
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                RandomCrop(config.patch_size),
                RandomFlip_LR(),
                RandomFlip_UD(),
                ToTensor()
            ])
        )
        return DataLoader(
            dst,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker
        )
    else:
        dst = dst_cls(
            task=args.task,
            split=split,
            is_val=True,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)


def make_model_all():
    model = VNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )

    return model, optimizer




class DistDW:
    def __init__(self, num_cls, do_bg=False, momentum=0.95):
        self.num_cls = num_cls
        self.do_bg = do_bg
        self.momentum = momentum

    def _cal_weights(self, num_each_class):
        num_each_class = torch.FloatTensor(num_each_class).cuda()
        P = (num_each_class.max()+1e-8) / (num_each_class+1e-8)
        P_log = torch.log(P)
        weight = P_log / P_log.max()
        return weight

    def init_weights(self, labeled_dataset):
        if labeled_dataset.unlabeled:
            raise ValueError
        num_each_class = np.zeros(self.num_cls)
        for data_id in labeled_dataset.ids_list:
            _, _, label = labeled_dataset._get_data(data_id)
            label = label.reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp
        weights = self._cal_weights(num_each_class)
        self.weights = weights * self.num_cls
        return self.weights.data.cpu().numpy()

    def get_ema_weights(self, pseudo_label):
        pseudo_label = torch.argmax(pseudo_label.detach(), dim=1, keepdim=True).long()
        label_numpy = pseudo_label.data.cpu().numpy()
        num_each_class = np.zeros(self.num_cls)
        for i in range(label_numpy.shape[0]):
            label = label_numpy[i].reshape(-1)
            tmp, _ = np.histogram(label, range(self.num_cls + 1))
            num_each_class += tmp

        cur_weights = self._cal_weights(num_each_class) * self.num_cls
        self.weights = EMA(cur_weights, self.weights, momentum=self.momentum)
        return self.weights



class DiffDW:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(config.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        return weights

    def cal_weights(self, pred,  label):
        x_onehot = torch.zeros(pred.shape).cuda()
        output = torch.argmax(pred, dim=1, keepdim=True).long()
        x_onehot.scatter_(1, output, 1)
        y_onehot = torch.zeros(pred.shape).cuda()
        y_onehot.scatter_(1, label, 1)
        cur_dice = self.dice_func(x_onehot, y_onehot, is_training=False)
        delta_dice = cur_dice - self.last_dice
        cur_cls_learn = torch.where(delta_dice>0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        cur_cls_unlearn = torch.where(delta_dice<=0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        self.last_dice = cur_dice
        self.cls_learn = EMA(cur_cls_learn, self.cls_learn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)
        cur_diff = torch.pow(cur_diff, 1/5)
        self.dice_weight = EMA(1. - cur_dice, self.dice_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        weights = cur_diff * self.dice_weight
        weights = weights / weights.max()
        return weights * self.num_cls

class Conf:
    def __init__(self, num_cls):
        self.conf_weight = torch.ones(num_cls).float().cuda() 
        self.conf_cur = torch.zeros(num_cls).float().cuda() + 1e-8
        self.iter = 0
    def ema(self,new_weight):
        self.conf_cur = self.conf_cur/(self.iter+1e-8)
        self.conf_weight = EMA(self.conf_cur, self.conf_weight, momentum=0.99)
        self.iter = 1
        return self.conf_weight
    def update(self,this_conf):
        self.iter += 1
        this_conf = (1-this_conf+1e-2)/((1-this_conf).max()+1e-2) 
        this_conf = torch.pow(this_conf,0.2)
        self.conf_cur += this_conf
        
        
    def get_conf(self):
        return self.conf_weight

if __name__ == '__main__':
    import random
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # make logger file
    snapshot_path = f'./logs/{args.exp}/'
    fold = str(args.exp[-1])

    if args.task == 'colon':
        #args.split_unlabeled = args.split_unlabeled+'_'+fold
        args.split_labeled = args.split_labeled+'_'+fold
        args.split_eval = args.split_eval+'_'+fold
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    # make logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # make data loader
    unlabeled_loader = make_loader(args.split_unlabeled, unlabeled=True)
    labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset))
    eval_loader = make_loader(args.split_eval, is_training=False)
    if args.task == 'colon':
        test_loader = make_loader(f'test_{fold}', is_training=False)
    else:
        test_loader = make_loader('test', is_training=False)

    logging.info(f'{len(labeled_loader)} itertations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} itertations per epoch (unlabeled)')

    # make model, optimizer, and lr scheduler
    model_A, optimizer_A = make_model_all()
    model_B, optimizer_B  = make_model_all()
    model_A = kaiming_normal_init_weight(model_A)
    model_B = kaiming_normal_init_weight(model_B)
    #model_B = xavier_normal_init_weight(model_B)
    ema_model_A = create_ema_model(config, model_A)
    ema_model_B = create_ema_model(config, model_B)

    ema_model_A.eval()
    ema_model_A = ema_model_A.cuda()


    ema_model_B.eval()
    ema_model_B = ema_model_B.cuda()
    
    
    # make loss function
    diffdw = DiffDW(config.num_cls, accumulate_iters=50)
    distdw = DistDW(config.num_cls, momentum=0.99)
    diffdw_mix = DiffDW(config.num_cls, accumulate_iters=50)

    conf = Conf(config.num_cls)
    print(conf.get_conf())

    weight_A = diffdw.init_weights()
    weight_B = distdw.init_weights(labeled_loader.dataset)

    loss_func_A     = make_loss_function(args.sup_loss, weight_A)
    loss_func_B     = make_loss_function(args.sup_loss, weight_B)
    cps_loss_func_A = make_loss_function(args.cps_loss, weight_A)
    cps_loss_func_B = make_loss_function(args.cps_loss, weight_B)

    loss_func_A_ema     = make_loss_function(args.sup_loss, weight_A)
    loss_func_B_ema     = make_loss_function(args.sup_loss, weight_B)
    cps_loss_func_A_ema = make_loss_function(args.cps_loss, weight_A)
    cps_loss_func_B_ema = make_loss_function(args.cps_loss, weight_B)

    if args.mixed_precision:
        amp_grad_scaler = GradScaler()

    cps_w = get_current_consistency_weight(0)
    best_eval = 0.0
    best_epoch = 0
    
    average_per_channel_total = torch.ones(config.num_cls).cuda()  # 假设通道数为config.num_cls
    batch_averages = torch.ones(config.num_cls).cuda()
    
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_cps_list = []
        loss_sup_list = []

        model_A.train()
        model_B.train()
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()

            image_l, label_l = fetch_data(batch_l)
            
            image_u = fetch_data(batch_u, labeled=False)
            
            
            image = torch.cat([image_l, image_u], dim=0)
            tmp_bs = image.shape[0] // 2

            label_l_backup = label_l.clone().squeeze(1).detach()
            conf_mask = torch.nn.functional.one_hot(label_l_backup, config.num_cls).permute(0,4,1,2,3).float()
            conf_mask[conf_mask!=0] = 1
            ones_per_channel = conf_mask.sum(dim=[0, 2, 3, 4]).detach().cuda()

            if args.mixed_precision:
                with autocast():
                    output_A = model_A(image)
                    output_B = model_B(image)
                    #del image

                    # sup (ce + dice)
                    output_A_l, output_A_u = output_A[:tmp_bs, ...], output_A[tmp_bs:, ...]
                    output_B_l, output_B_u = output_B[:tmp_bs, ...], output_B[tmp_bs:, ...]

                    conf_map_A =  F.softmax(output_A_l) * conf_mask #bcdwh
                    sum_per_channel = conf_map_A.sum(dim=[0, 2, 3, 4])
                    conf_weight = (sum_per_channel.detach()+1) / (ones_per_channel.detach()+1)
                    
                    conf.update(conf_weight)
                    #batch_averages = conf_weight.detach()
                    #conf_weight =1-( conf_weight / conf_weight.max())
                    conf_weight = conf.get_conf()
                    #conf_weight = (1-conf_weight+1e-2)/((1-conf_weight).max()+1e-2)  #1/torch.exp(conf_weight) #
                    #print(conf_weight)
                    #conf_weight = torch.sqrt(1-conf_weight)
                    #print(f'conf_weight: {conf_weight}')
                    # cps (ce only)
                    max_A = torch.argmax(output_A.detach(), dim=1, keepdim=True).long()
                    max_B = torch.argmax(output_B.detach(), dim=1, keepdim=True).long()


                    weight_A = diffdw.cal_weights(output_A_l.detach(), label_l.detach())
                    weight_B = distdw.get_ema_weights(output_B_u.detach())



                    loss_func_A.update_weight(weight_A*conf_weight)
                    loss_func_B.update_weight(weight_B)
                    cps_loss_func_A.update_weight(weight_A*conf_weight)
                    cps_loss_func_B.update_weight(weight_B)
                    
                    
                    if epoch_num>0 and random.random() <= 1:
                        #print('DMF')
                        diffdw.accumulate_iters = diffdw.accumulate_iters
                        output_A_ema = ema_model_A(image)
                        output_B_ema = ema_model_B(image)
                        max_A_ema = torch.argmax(output_A_ema.detach(), dim=1, keepdim=True).long()
                        max_B_ema = torch.argmax(output_B_ema.detach(), dim=1, keepdim=True).long()

                        label_mixed_pre_A = torch.cat([label_l, max_A_ema[tmp_bs:, ...]], dim=0)
                        label_mixed_pre_B = torch.cat([label_l, max_B_ema[tmp_bs:, ...]], dim=0)

                        for image_i in range(image.shape[0]):

                            ignore_label = None #remove background
                            classes_A = torch.unique(max_A_ema[image_i])
                            weight_A_mix = (weight_A * conf_weight) * config.num_cls
                            if ignore_label:

                                classes_A = classes_A[classes_A != ignore_label]
                            w_A = weight_A_mix[classes_A.detach().tolist()]    

                            unique_conf_dist = torch.distributions.categorical.Categorical(probs = w_A)
                            labels_select = torch.unique(unique_conf_dist.sample(sample_shape=[max(int(len(classes_A)/1),1)]))
                            classes_A = labels_select.cuda()
                            #torch.save(max_A.detach().cpu(), 'max_a.pth')                    
                            mask_A = (max_A.unsqueeze(-1) == labels_select).any(-1).long()


                            classes_B = torch.unique(max_B_ema[image_i])
                            if ignore_label:

                                classes_B = classes_B[classes_B != ignore_label]
                            w_B = weight_B[classes_B.detach().tolist()]    

                            unique_conf_dist = torch.distributions.categorical.Categorical(probs = w_B)
                            labels_select = torch.unique(unique_conf_dist.sample(sample_shape=[max(int(len(classes_B)/1),1)]))
                            classes_B = labels_select.cuda()
                            #torch.save(max_B.detach().cpu(), 'max_a.pth')
                            mask_B = (max_B.unsqueeze(-1) == labels_select).any(-1).long()



                        mixed_A,mixed_A_label = mix(mask_A, data=image, target=label_mixed_pre_A)
                        mixed_B,mixed_B_label = mix(mask_B, data=image, target=label_mixed_pre_B)
                        mixed_output_A = model_A(mixed_B)
                        mixed_output_B = model_B(mixed_A)

                        weight_A_ema = diffdw_mix.cal_weights(mixed_output_A.detach(), label_mixed_pre_A.detach())
                        weight_B_ema = distdw.get_ema_weights(mixed_B_label.detach())

                        max_A_ema = torch.argmax(mixed_output_A.detach(), dim=1, keepdim=True).long()
                        max_B_ema = torch.argmax(mixed_output_B.detach(), dim=1, keepdim=True).long()
                        

                        loss_sup = (1.5*loss_func_A(output_A_l, label_l) +  \
                            loss_func_B(output_B_l, label_l)*1.5 + \
                            loss_func_A_ema(mixed_output_A, max_A_ema)*0.5 + \
                            loss_func_B_ema(mixed_output_B, max_B_ema)*0.5)/2

                       
                        del image,mask_A,mixed_A,mixed_A_label,label_mixed_pre_A,mask_B,mixed_B,mixed_B_label,label_mixed_pre_B,label_l

                    else:                    


                        loss_sup = loss_func_A(output_A_l, label_l) + loss_func_B(output_B_l, label_l)


                    #print(f'label_l: {label_l.shape}, max : {label_l.max()}, max_A: {max_A.shape}, max_A: {max_A.max()}')
                    #loss_sup = loss_func_A(output_A_l, label_l) + loss_func_B(output_B_l, label_l)
                    loss_cps = cps_loss_func_A(output_A, max_B) + cps_loss_func_B(output_B, max_A)
                    loss = loss_sup + cps_w * loss_cps



                # backward passes should not be under autocast.
                amp_grad_scaler.scale(loss).backward()
                amp_grad_scaler.step(optimizer_A)
                amp_grad_scaler.step(optimizer_B)
                amp_grad_scaler.update()
                # if epoch_num>0:

            else:
                raise NotImplementedError

            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_cps_list.append(loss_cps.item())
        conf.ema(conf_weight)
        writer.add_scalar('lr', get_lr(optimizer_A), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/sup', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/cps', np.mean(loss_cps_list), epoch_num)
        # print(dict(zip([i for i in range(config.num_cls)] ,print_func(weight_A))))
        writer.add_scalars('class_weights/A', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_A))), epoch_num)
        writer.add_scalars('class_weights/B', dict(zip([str(i) for i in range(config.num_cls)] ,print_func(weight_B))), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list)}')
        # logging.info(f'     cps_w: {cps_w}')
        # if epoch_num>0:
        logging.info(f"     Class Weights A: {print_func(weight_A)}, lr: {get_lr(optimizer_A)}")
        logging.info(f"     Class Weights B: {print_func(weight_B)}")
        logging.info(f"     Class Conf     : {print_func(conf.get_conf())}")
        # logging.info(f"     Class Weights u: {print_func(weight_u)}")
        # lr_scheduler_A.step()
        # lr_scheduler_B.step()
        optimizer_A.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_B.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        # print(optimizer_A.param_groups[0]['lr'])
        cps_w = get_current_consistency_weight(epoch_num)
        '''
        if config.task == 'colon':
            args.start_mix = 10
        args.start_mix = 50   
        '''
        if epoch_num % 10 == 0 or epoch_num >=50:

            # ''' ===== evaluation
            dice_list = [[] for _ in range(config.num_cls-1)]
            model_A.eval()
            model_B.eval()
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    output = (model_A(image) + model_B(image))/2.0
                    # output = model_B(image)
                    del image

                    shp = output.shape
                    gt = gt.long()
                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)

                    x_onehot = torch.zeros(shp).cuda()
                    output = torch.argmax(output, dim=1, keepdim=True).long()
                    x_onehot.scatter_(1, output, 1)


                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)

            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save({
                    'A': model_A.state_dict(),
                    'B': model_B.state_dict()
                }, save_path)
                
                logging.info(f'saving best model to {save_path}')
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
            # '''
        config.early_stop_patience = 50
        if epoch_num - best_epoch == config.early_stop_patience:
            
            logging.info(f'Early stop.')
            break
            
           

    writer.close()
