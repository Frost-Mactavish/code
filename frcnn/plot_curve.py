import json
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detection import transform_
import utils.distributed_utils as utils
from dataset import DIORDataset
from detection.coco_eval import CocoEvaluator
from detection.coco_utils import get_coco_api_from_dataset

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with open('config.json') as f:
    config = json.load(f)


@torch.no_grad()
def evaluate(model, dataloader):
    ''' evaluate performance of detector using coco api '''
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(dataloader.dataset)
    iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(dataloader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    pr_array = coco_evaluator.coco_eval[iou_types[0]].eval['precision']

    return pr_array

def load_weights(weight_path, num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    weights_dict = torch.load(weight_path,map_location='cpu', weights_only=True)
    weights_dict = weights_dict['model'] if 'model' in weights_dict else weights_dict
    model.load_state_dict(weights_dict)

    return model


def load_cowc(weight, img_dir):
    mean, std = config['cowc']['mean'], config['cowc']['std']
    anno_dir = config['cowc']['test_dir']['anno']

    model = load_weights(weight, 2).to(device)

    data_transform = transform_.Compose([
                transform_.ToTensor(),
                transform_.Normalize(mean, std)
            ])
    dataloader = DataLoader(
        COWCDataset(img_dir=img_dir,
                    anno_dir=anno_dir,
                    transform=data_transform),
        collate_fn=collate_fn
    )

    array = evaluate(model=model, dataloader=dataloader)
    array = array[0, :, 0, 0, 2]

    return array


def load_aitod(weight, img_dir):
    mean, std = config['aitod']['mean'], config['aitod']['std']
    anno_dir = config['aitod']['json_path']

    model = load_weights(weight, 9).to(device)

    data_transform = transform_.Compose([
                transform_.ToTensor(),
                transform_.Normalize(mean, std)
            ])
    dataloader = DataLoader(
        AitodDataset(img_dir=img_dir,
                     json_path=anno_dir,
                     transform=data_transform,
                     val=True),
        collate_fn=collate_fn
    )
    array = evaluate(model=model, dataloader=dataloader)
    array = np.mean(array[0, :, :, 0, 2], axis=1)

    return array


def plot_ap():
    ''' plot AP-IoU curve for COWC and AI-TOD dataset '''

    # mAP at IoU=0.5:0.95, acquired from pycocotools.COCOEvaluator
    aitod_HR = np.array([45.3, 42.5, 39.9, 37.0, 33.3, 28.1, 21.2, 16.2, 4.3, 3.7])
    aitod_LR = np.array([18.4, 16.0, 13.0, 9.0, 6.4, 3.8, 1.7, 0.8, 0.0, 0.0])
    aitod_ISR = np.array([43.3, 41.0, 37.4, 31.6, 26.6, 17.9, 11.4, 6.1, 0.5, 0.2])
    aitod_SR = np.array([47.9, 44.9, 41.5, 36.7, 28.9, 22.3, 14.7, 8.8, 1.5, 0.3])

    cowc_HR = np.array([94.1, 93.1, 92.8, 91.3, 88.3, 83.8, 73.9, 59.7, 12.5, 8.7])
    cowc_LR = np.array([83.3, 79.4, 73.6, 63.6, 48.4, 33.2, 17.6, 6.5, 1.7, 0.1])
    cowc_ISR = np.array([93.3, 92.4, 91.9, 89.2, 85.2, 77.4, 66.8, 45.2, 3.9, 2.2])
    cowc_SR = np.array([91.5, 91.4, 90.9, 89.6, 85.8, 79.9, 71.4, 54.9, 10.3, 6.8])

    # plot AP-IoU curve, subplot 1 for cowc, subplot 2 for aitod
    x = np.arange(50, 100, 5)
    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plt.title('COWC数据集')
    y = np.arange(0, 101, 20)
    plt.plot(x, cowc_HR, 'r', label='FRCNN(HR-HR)', linewidth=1)
    plt.plot(x, cowc_SR, 'b', label='我们的方法 + FRCNN(SR-SR)', linewidth=1)
    plt.plot(x, cowc_ISR, 'g', label='ESRGAN + FRCNN(SR-SR)', linewidth=1)
    plt.xlabel('IoU')
    plt.ylabel('AP')
    plt.xticks(x, x / 100)
    plt.yticks(y, ['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.xlim(50, 95)
    plt.ylim(0, 100)
    plt.legend(loc='lower left')
    plt.grid()

    plt.subplot(122)
    plt.title('AI-TOD数据集')
    y = np.arange(0, 51, 10)
    plt.plot(x, aitod_HR, 'r', label='FRCNN(HR-HR)', linewidth=1)
    plt.plot(x, aitod_SR, 'b', label='我们的方法 + FRCNN(SR-SR)', linewidth=1)
    plt.plot(x, aitod_ISR, 'g', label='ESRGAN + FRCNN(SR-SR)', linewidth=1)
    plt.xlabel('IoU')
    plt.ylabel('AP')
    plt.xticks(x, x / 100)
    plt.yticks(y, ['0%', '10%', '20%', '30%', '40%', '50%'])
    plt.xlim(50, 95)
    plt.ylim(0, 50)
    plt.legend(loc='lower left')
    plt.grid()

    plt.savefig('AP-IoU.svg')
    plt.show()


    # plot AP-IoU curve(cmp between SR and LR), subplot 1 for cowc, subplot 2 for aitod
    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plt.title('COWC数据集')
    y = np.arange(0, 101, 20)
    plt.plot(x, cowc_SR, 'b', label='我们的方法 + FRCNN(SR-SR)', linewidth=1)
    plt.plot(x, cowc_LR, 'r', label='FRCNN(LR-LR)', linewidth=1)
    plt.xlabel('IoU')
    plt.ylabel('AP')
    plt.xticks(x, x / 100)
    plt.yticks(y, ['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.xlim(50, 95)
    plt.ylim(0, 100)
    plt.legend(loc='lower left')
    plt.grid()

    plt.subplot(122)
    plt.title('AI-TOD数据集')
    y = np.arange(0, 51, 10)
    plt.plot(x, aitod_SR, 'b', label='我们的方法 + FRCNN(SR-SR)', linewidth=1)
    plt.plot(x, aitod_LR, 'r', label='FRCNN(LR-LR)', linewidth=1)
    plt.xlabel('IoU')
    plt.ylabel('AP')
    plt.xticks(x, x / 100)
    plt.yticks(y, ['0%', '10%', '20%', '30%', '40%', '50%'])
    plt.xlim(50, 95)
    plt.ylim(0, 50)
    plt.legend(loc='upper right')
    plt.grid()

    plt.savefig('AP-IoU-1.svg')
    plt.show()


def plot_pr():
    ''' plot Precision-Recall curve for COWC and AI-TOD dataset '''

    # get pr array for cowc
    hr_dir = config['cowc']['test_dir']['HR']
    sr_dir = config['cowc']['test_dir']['ISR']
    isr_dir = config['cowc']['test_dir']['SR']
    sr_no_dir = '../dataset/COWC/test/SR(no_edgecst)'

    hr_weight = 'saved_weights/cowc/cowc-HR.pth'
    sr_weight = 'saved_weights/cowc/cowc-ISR.pth'
    isr_weight = 'saved_weights/cowc/cowc-SR.pth'
    sr_no_weight = 'saved_weights/cowc/cowc-SR(no_edgecst).pth'

    cowc_hr_array = load_cowc(hr_weight, hr_dir, anno_dir)
    cowc_isr_array = load_cowc(isr_weight, isr_dir, anno_dir)
    cowc_sr_array = load_cowc(sr_weight, sr_dir, anno_dir)
    cowc_sr_no_array = load_cowc(sr_no_weight, sr_no_dir, anno_dir)


    # get pr array for aitod

    hr_dir = config['aitod']['img_dir']['HR']
    sr_dir = config['aitod']['img_dir']['SR']
    isr_dir = config['aitod']['img_dir']['ISR']

    hr_weight = 'saved_weights/aitod/aitod-HR.pth'
    sr_weight = 'saved_weights/aitod/aitod-SR.pth'
    isr_weight = 'saved_weights/aitod/aitod-ISR.pth'

    aitod_hr_array = load_aitod(hr_weight, hr_dir, anno_dir)
    aitod_sr_array = load_aitod(sr_weight, sr_dir, anno_dir)
    aitod_isr_array = load_aitod(isr_weight, isr_dir, anno_dir)

    # plot pr curve , subplot 1 for cowc, subplot 2 for aitod
    x = np.arange(0.0, 1.01, 0.01)
    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plt.plot(x, cowc_hr_array, 'r', linewidth=1, label='FRCNN(HR-HR)')
    plt.plot(x, cowc_sr_array, 'b', linewidth=1, label='我们的方法 + FRCNN(SR-SR)')
    plt.plot(x, cowc_isr_array, 'g', linewidth=1, label='ESRGAN + FRCNN(SR-SR)')
    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.title('COWC数据集')
    plt.grid()
    plt.legend(loc='lower left')

    plt.subplot(122)
    plt.plot(x, aitod_hr_array, 'r', linewidth=1, label='FRCNN(HR-HR)')
    plt.plot(x, aitod_sr_array, 'b', linewidth=1, label='我们的方法 + FRCNN(SR-SR)')
    plt.plot(x, aitod_isr_array, 'g', linewidth=1, label='ESRGAN + FRCNN(SR-SR)')
    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.title('AI-TOD数据集')
    plt.grid()
    plt.legend(loc='lower left')
    plt.savefig('pr.svg')
    plt.show()


def plot_cst():
    '''
    compare the AP-IoU curve and Precision-Recall curve
    for our method with and without edge_cst loss
    '''

    # mAP at IoU=0.5:0.95, acquired from pycocotools.COCOEvaluator
    cowc_SR = np.array([91.5, 91.4, 90.9, 89.6, 85.8, 79.9, 71.4, 54.9, 10.3, 6.8])
    cowc_SR_no = np.array([92.4, 91.3, 89.9, 85.0, 77.7, 65.0, 47.4, 2.5, 2.0, 1.9])

    # get pr array for cowc
    sr_dir = config['cowc']['test_dir']['ISR']
    sr_no_dir = '../dataset/COWC/test/SR(no_edgecst)'

    sr_weight = 'saved_weights/cowc/cowc-ISR.pth'
    sr_no_weight = 'saved_weights/cowc/cowc-SR(no_edgecst).pth'

    cowc_sr_array = load_cowc(sr_weight, sr_dir)
    cowc_sr_no_array = load_cowc(sr_no_weight, sr_no_dir)


    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    x = np.arange(50, 100, 5)
    y = np.arange(0, 101, 20)
    plt.plot(x, cowc_SR, 'b', label='使用边缘一致性损失', linewidth=1)
    plt.plot(x, cowc_SR_no, 'r', label='未使用边缘一致性损失', linewidth=1)
    plt.xlabel('IoU')
    plt.ylabel('AP')
    plt.xticks(x, x / 100)
    plt.yticks(y, ['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.xlim(50, 95)
    plt.ylim(0, 100)
    plt.legend(loc='lower left')
    plt.title('AP-IoU曲线')
    plt.grid()

    plt.subplot(122)
    x = np.arange(0.0, 1.01, 0.01)
    plt.plot(x, cowc_sr_array, 'b', linewidth=1, label='使用边缘一致性损失')
    plt.plot(x, cowc_sr_no_array, 'r', linewidth=1, label='未使用边缘一致性损失')
    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.title('Precision-Recall曲线')
    plt.grid()
    plt.legend(loc='lower left')
    plt.savefig('edge_cst.svg')
    plt.show()


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['simsun']
    plt.rcParams['axes.unicode_minus'] = False

    plot_ap()
    plot_pr()
    plot_cst()




