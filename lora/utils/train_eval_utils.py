import os
import sys
import math
import json
import time
from optparse import Option
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageColor, ImageDraw, ImageFont

import torch
import torchvision

import utils.distributed_utils as utils
from detection.coco_eval import CocoEvaluator


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def train_one_epoch(model,
                    optimizer,
                    dataloader,
                    device,
                    epoch,
                    print_freq: Optional[int] = 100):
    '''
    define the training procedure for one epoch
    
    Args:
        print_freq (int, optional): interval for printing info

    Returns:
        avg_loss (float): averaged overall loss over batches
        loss_dict (dict): loss dict of different components, including rpn and roi_head
        now_lr (float): learning rate of current epoch
    '''

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    header = 'Epoch: [{}]'.format(epoch)

    # average loss over batches in one epoch
    loss = {'classifier': 0.0, 'box_reg': 0.0, 'objectness': 0.0, 'rpn_box_reg': 0.0}
    for i, [images, targets] in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]

        loss_dict = model(images, targets)

        # average 4 losses over batches
        for loss_name, batch_loss in zip(loss.keys(), loss_dict.values()):
            batch_loss = batch_loss.item()
            if not math.isfinite(batch_loss):
                print(f"Loss is {batch_loss:.4f}, training terminate")
                sys.exit(1)

            loss[loss_name] = (loss[loss_name] * i + batch_loss) / (i + 1) # update mean losses

        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses, **loss)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    aggregated_loss = sum(loss.values())

    return aggregated_loss, loss, now_lr


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


@torch.no_grad()
def evaluate(model,
             dataloader,
             coco_gt,
             device,
             phase: str,
             print_feq: Optional[int] = 100):
    '''
    define procedure of test process

    Args:
        coco_gt (Object): Ground organized conforming to coco api
        phase (str): base, inc or joint
        print_feq (int, optional): interval for printing logs

    Returns:
        coco_info (list): list of evalution stats
    '''
    cpu_device = torch.device("cpu")
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco_gt, iou_types)

    from dataset import get_class_dict
    class_dict = get_class_dict('DIOR', phase)
    category_index = {v: k for k, v in class_dict.items()}

    for i, [image, targets] in enumerate(metric_logger.log_every(dataloader, print_feq, header)):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model.eval()
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

    # pycocotools calc AvergePrecision for each class
    # ------------------------------------------------
    coco_eval = coco_evaluator.coco_eval["bbox"]
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_eval)

    # calculate voc info for every classes(IoU=0.5)
    voc_map_info_list = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list.append(" {}: {:.2f}".format(category_index[i + 1], stats[1]*100))

    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)

    # ------------------------------------------------
    # end of calc AP for each class

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return coco_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def tensor2img(tensor, dataset, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    #tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    #tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3: # Now only works for dimension 3
        img_np = tensor.squeeze().cpu().numpy().transpose(1,2,0).copy()
        mean = np.array([0.3856, 0.3973, 0.3619])
        std = np.array([0.1998, 0.1853, 0.1866])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def nms(bounding_boxes, confidence_score, threshold):
    ''' filter out undesired bboxes due to a loose threshold of the NMS in frcnn implemented by PyTorch '''
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score


def draw_text(draw,
              box: list,
              cls: int,
              score: float,
              category_index: dict,
              color: str,
              font_size: int = 24):
    """
    将目标边界框和类别信息绘制到图片上
    """

    x0, y0, x1, y1 = box
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.


def draw_boxes(img_path, pred_boxes, pred_classes, pred_score, GT_boxes, GT_classes,
               category_index, threshold=0.5, line_thickness=6, font_size=24):
    '''
    pred_list: bbox coordinates, confidence, label wrapped together for each prediction respectively
    '''

    predict = Image.open(img_path)
    GT = predict.copy()

    font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf', font_size)

    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in pred_classes]

    draw = ImageDraw.Draw(predict)
    for box, cls, score, color in zip(pred_boxes, pred_classes, pred_score, colors):
        if score < threshold:
            continue
        x0, y0, x1, y1 = box
        # 绘制目标边界框
        draw.rectangle([x0, y0, x1, y1], width=line_thickness, outline=color)
        # 绘制类别和概率信息
        # text = f"{category_index[str(int(cls))]}: {int(100 * float(score))}%"
        # draw.text((x0, y0), text, font=font, fill='White', font_size=font_size)

    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in GT_classes]

    draw = ImageDraw.Draw(GT)
    for box, cls, color in zip(GT_boxes, GT_classes, colors):
        x0, y0, x1, y1 = box
        # 绘制目标边界框
        draw.rectangle([x0, y0, x1, y1], width=line_thickness, outline=color)
        # 绘制类别和概率信息
        # text = f"{category_index[str(int(cls))]}%"
        # draw.text((x0, y0), text, font=font, fill='White', font_size=font_size)


    return predict, GT


def inference(model, dataloader, threshold, device, srcdir):
    '''
    threshold: IoU threshold for NMS
    savedir: path to save imgs with bbox
    srcdir: path to load imgs
    '''
    print('Start Inference')

    with open('classes.json', 'r') as f:
        class_dict = json.load(f)
    category_index = {str(v):str(k) for k,v in class_dict.items()}

    model.eval()
    with torch.no_grad():
        for img, target in dataloader:
            img = [I.to(device) for I in img]
            file_name = [t['filename'] for t in target]

            img_path = [os.path.join(srcdir, name) for name in file_name]
            GT_boxes = [t['boxes'].numpy() for t in target]
            GT_classes = [t['labels'].numpy() for t in target]

            outputs = model(img)

            pred_score = [o['scores'].to('cpu').numpy() for o in outputs]
            pred_boxes = [o['boxes'].to('cpu').numpy() for o in outputs]
            pred_classes = [o['labels'].to('cpu').numpy() for o in outputs]

            # pred_boxes, pred_score = nms(pred_boxes, pred_score, threshold)

            draw_queue = [{'pred_boxes':a, 'pred_classes':b, 'pred_score':c, 'GT_boxes':d, 'GT_classes':e, 'img_path':f}
                          for a,b,c,d,e,f in zip(pred_boxes, pred_classes, pred_score, GT_boxes, GT_classes, img_path)]

            for element in draw_queue:
                img_path = element['img_path']
                pred_boxes = element['pred_boxes']
                pred_classes = element['pred_classes']
                pred_score = element['pred_score']

                GT_boxes = element['GT_boxes']
                GT_classes = element['GT_classes']

                predict, GT = draw_boxes(img_path, pred_boxes, pred_classes, pred_score,
                                         GT_boxes, GT_classes, category_index, threshold)

                plt.imshow(predict)
                plt.show()

                file_name = os.path.basename(img_path)
                predict.save('{}-predict.jpg'.format(os.path.basename(file_name)))
                GT.save('{}-GT.jpg'.format(os.path.basename(file_name)))

            break

    print('Finish Inference')


