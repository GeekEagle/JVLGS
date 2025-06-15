import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from VLJGS.lib.main_vljgs import VLJGS
from dataloaders import kfold_dataloader    
from torchvision import transforms
import cv2
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='SimGas') 
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--trainsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./pretrain/SimGas_trained/simgas_0_best.pth')
parser.add_argument('--pretrained_cod10k', default='./pretrain/cod10k_encoder.pth', help='path to the pretrained Resnet')
parser.add_argument('--cuda', type=str, default='cuda:0', help='use cuda')
parser.add_argument('--kernel_size', type=int, default=9, help='kernel size for morphological operation')
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
parser.add_argument('--text', type=str, default=["white steam", "floating steam", "billowing smoke", "flowing smoke"], help='text input')
# parser.add_argument('--text', type=str, default=["", "", "", "flowing smoke"], help='text input')
opt = parser.parse_args()

def show_pictures(res, name, image_path, save_path, vis_path):
    res_pil = transforms.ToPILImage()(res)
    name = name.replace('jpg', 'png')
    frame_path = os.path.join(save_path, name)
    res_pil.save(frame_path)
    vis_file = os.path.join(vis_path, name)
    image_array = cv2.imread(image_path)
    empty_mask = np.zeros_like(image_array)
    empty_mask[:,:, 1] = res
    blend = (image_array/255 + empty_mask/255)*0.8
    blend = np.where(blend > 1, 1, blend)
    blend = (blend * 255).astype(np.uint8)  
    cv2.imwrite(vis_file, blend)

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
   
if __name__ == '__main__':
    j_sum = 0
    f1_total = 0
    jf_sum = 0
    save_root = './res/'
    for iter in range(5):
        _, test_loader, _, _ = kfold_dataloader(opt, iter)
        model = VLJGS(opt).to(opt.cuda)
        pth_path = f'pretrain/SimGas_trained/simgas_{iter}_best.pth'
        
        model.load_state_dict(torch.load(pth_path, map_location=opt.cuda))
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("google/owlv2-base-patch16")
        iou_sum = 0
        f1_sum = 0
        total_pictures = 0
        with torch.no_grad():
            for i, data_blob in enumerate(tqdm(test_loader)):
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
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opt.kernel_size, opt.kernel_size))
                    prediction = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel)
            
                    groundtruth = groundtruth.squeeze()
                    intersection, union = compute_iou(prediction, groundtruth)
                    if union > 0:
                        total_pictures += 1
                        f1_sum += f1_score(groundtruth, prediction)
                        iou_sum += intersection / union
                
                    prediction = (prediction * 255)
                    save_path = os.path.join(save_root, scene[j], 'masks')
                    vis_path = os.path.join(save_root, scene[j], 'vis')
                    if not os.path.exists(save_path):
                       os.makedirs(save_path)
                       os.makedirs(vis_path)
                    show_pictures(prediction, names[0][j], image_path[0][j], save_path, vis_path)
        
            f1 = f1_sum / total_pictures
            Jaccard = iou_sum / total_pictures
            j_f = (f1 + Jaccard) / 2
        
            j_sum += Jaccard
            f1_total += f1
            jf_sum += j_f
            print('iter:{}, Jaccard: {}, f1: {}, J_F:{}.'.format(iter, Jaccard, f1, j_f))
    print(f'Jaccard: {j_sum/5}, f1: {f1_total/5}, J_F:{jf_sum/5}.')
