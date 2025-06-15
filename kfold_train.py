from __future__ import print_function, division
import sys
# sys.path.append('lib')
sys.path.append('dataloaders')

import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
from datetime import datetime
import wandb
import cv2, random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer
from lib.main_vljgs import VLJGS
from dataloaders import kfold_dataloader
from utils.pyt_utils import load_model
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def train(train_loader, model, optimizer, epoch, save_path, iter, tokenizer):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for j, data_blob in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            image, texts, gts, name, scene, image_path = data_blob
            text = [describe[0] for describe in texts]
            text_encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            text = {k: v.to(opt.cuda) for k, v in text_encoded.items()}
            image = [img.to(opt.cuda) for img in image]
            gts = gts.to(opt.cuda)
            preds1, preds2, pred = model(image, text)
            
            loss = structure_loss(pred, gts)
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data

            if j % 500 == 0 or j == total_step or j == 0:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.format(datetime.now(), epoch, opt.epoch, j, total_step, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}'.format(epoch, opt.epoch, j, total_step, loss.data))
                writer.add_scalars('Loss_Statistics',{'Loss_total': loss.data},global_step=step)

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + f'/{opt.dataset}_{iter}_{epoch}.pth')
        raise

def val(test_loader, model, epoch, save_path, tokenizer, iter):
    """
    validation function
    """
    global best_epoch, best_iou
    model.eval()
    with torch.no_grad():
        iou_sum = 0
        f1_sum = 0
        total_pictures = 0
        
        for i, data_blob in enumerate(test_loader, start=1):
            images, texts, gt, scene, names, image_path = data_blob
            text = [describe[0] for describe in texts]
            text_encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            text = {k: v.to(opt.cuda) for k, v in text_encoded.items()}
            image = [img.to(opt.cuda) for img in images]
       
            _, _, pred = model(image, text)
            image_array = cv2.imread(image_path[0][0])

            for j in range(0, len(pred)):
                per_perd = pred[j].unsqueeze(0)
                res = F.upsample(per_perd, size=image_array.shape[:2], mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                prediction = np.where(res > 0.5, 1, 0)
                groundtruth = np.where(gt[j] > 0.5, 1, 0)
                
                prediction = prediction.astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                prediction = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel)
                
                intersection, union = compute_iou(prediction, groundtruth)  
                if union > 0:
                    total_pictures += 1
                    iou_sum += intersection / union
                    f1_sum += f1_score(groundtruth, prediction)
            
            prediction = prediction.squeeze()
            prediction = (prediction * 255).astype(np.uint8)
            groundtruth = (groundtruth * 255).astype(np.uint8)
            image = image_array.astype(np.uint8)
            random_number = random.randint(0, 100)
            if random_number >=98:
                wandb.log({"image": wandb.Image(image),
                           "prediction": wandb.Image(prediction),
                           "ground_truth": wandb.Image(groundtruth)
                           })
       
        Jaccard = iou_sum / total_pictures
        f1 = f1_sum / total_pictures
        j_f = (Jaccard + f1) / 2
        wandb.log({"Jaccard": Jaccard,
                   "F1": f1,
                   "Jaccard_F1": j_f,
                   })
        logging.info(f'Epoch: {epoch}, IOU: {Jaccard}, F1: {f1}, j_f: {j_f}.')
        if epoch == 1:
            best_iou = Jaccard
        else:
            if Jaccard > best_iou:
                best_epoch = epoch
                best_iou = Jaccard
                torch.save(model.state_dict(), save_path + f'/{opt.dataset}_{iter}_best.pth')
                print('Save state_dict successfully! Best epoch:{} in iter {}.'.format(epoch, iter))
        logging.info('[Val Info]:Epoch:{} bestEpoch:{} bestiou: {}'.format(epoch, best_epoch, best_iou))

def freeze_network(model):
    for name, p in model.named_parameters():
        if "fusion_conv" not in name:
            p.requires_grad = False

def compute_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection, union

def f1_score(groundtruth, prediction):
    groundtruth = groundtruth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.logical_and(groundtruth, prediction)
    recall = intersection.sum() / (groundtruth.sum() + 1e-8)
    precision = intersection.sum() / (prediction.sum() +1e-8)
    f1 = (2 * recall * precision) / (recall + precision + 1e-8)
    return f1

def test(test_loader, model, para_path, iter):
    if para_path is not None:
        model.load_state_dict(torch.load(para_path))
    model.eval()
    with torch.no_grad():
        iou_sum = 0
        f1_sum = 0
        total_pictures = 0
        
        for i, data_blob in enumerate(test_loader, start=1):
            images, texts, gt, scene, names, image_path = data_blob
            text = [describe[0] for describe in texts]
            text_encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            text = {k: v.to(opt.cuda) for k, v in text_encoded.items()}
            image = [img.to(opt.cuda) for img in images]
            _, _, pred = model(image, text)
            image_array = cv2.imread(image_path[0][0])

            for j in range(0, len(pred)):
                per_perd = pred[j].unsqueeze(0)
                res = F.upsample(per_perd, size=image_array.shape[:2], mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                prediction = np.where(res > 0.5, 1, 0)
                groundtruth = np.where(gt[j] > 0.5, 1, 0)
                
                prediction = prediction.astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                prediction = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel)
                
                groundtruth = groundtruth.squeeze()
                intersection, union = compute_iou(prediction, groundtruth)
                if union > 0:
                    total_pictures += 1
                    f1_sum += f1_score(groundtruth, prediction)
                    iou_sum += intersection / union
        
        f1 = f1_sum / total_pictures
        Jaccard = iou_sum / total_pictures
        j_f = (f1 + Jaccard) / 2
        logging.info('iter:{}, Jaccard: {}, f1: {}, J_F:{}.'.format(iter, Jaccard, f1, j_f))
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=60, help='epoch number')
    parser.add_argument('--trainsplit', type=str, default='train', help='train from checkpoints')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=6, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=45, help='every n epochs decay learning rate')
    parser.add_argument('--pretrained_cod10k', default='./pretrain/cod10k_encoder.pth', help='path to the pretrained Swin Transformer')
    parser.add_argument('--resume', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--cuda', type=str, default='cuda:0', help='use cuda? Default=True')
    parser.add_argument('--seed', type=int, default=2025, help='random seed to use. Default=123')
    parser.add_argument('--dataset',  type=str, default='SimGas', help='dataset name')
    parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
    parser.add_argument('--save_path', type=str,default='./pretrain/simgas_exp/',help='the path to save model and log')
    parser.add_argument('--valonly', action='store_true', default=False, help='skip training during training')
    parser.add_argument('--text', type=str, default=["white steam", "floating steam", "billowing smoke", "flowing smoke"], help='text input')
    opt = parser.parse_args()

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('===> Loading datasets')
    run = wandb.init(project='vljgs', name='training', config=opt)
    for iter in range(5):
        model = VLJGS(opt).to(opt.cuda)
        if opt.resume is not None and iter == 0:
            model = load_model(model=model, model_file=opt.resume, is_restore=True)
            print('Loading state dict from: {0}'.format(opt.resume))
        else:
            print("Cannot find model file at {}".format(opt.resume))

        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        tokenizer = AutoTokenizer.from_pretrained("google/owlv2-base-patch16")
        
        train_loader, val_loader, train_list, val_list = kfold_dataloader(opt, iter)
        total_step = len(train_loader)
        logging.info('{} iteration, train set:{}, val set:{}'.format(iter, train_list, val_list))

        step = 0
        writer = SummaryWriter(save_path + 'summary')
        best_epoch = 0
        best_iou = 0  

        logging.info("Start train...{}".format(iter))
        # val(val_loader, model, 0, save_path, tokenizer, iter)
        for epoch in range(1, opt.epoch+1):
            cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
            train(train_loader, model, optimizer, epoch, save_path, iter, tokenizer)
            if epoch % 3 == 0:
                val(val_loader, model, epoch, save_path, tokenizer, iter)
                
        logging.info('{} times training finished.'.format(iter))
        para_path =  save_path +f'/{opt.dataset}_{iter}_best.pth'
        test(val_loader, model, para_path, iter)