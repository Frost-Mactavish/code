import os
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageColor, ImageDraw

import torch
from torchvision import transforms

from dior_test import load_weights
from utils.train_eval_utils import STANDARD_COLORS


with open('config.json') as f:
    config = json.load(f)["DIOR"]


def main():

    root = config['root']
    mean, std = config['mean'], config['std']
    batch_size = config['test_batchSize']
    model_savedir = config['savedir']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_weights(os.path.join(model_savedir, 'DIOR-15-47.010.pth')).to(device)

    with open(classes.json, 'r') as f:
        class_dict = json.load(f)["DIOR"]
    category_index = {str(v): str(k) for k, v in class_dict.items()}

    original_img = Image.open(root + "/test_fog/16925.jpg")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        predictions = model(img.to(device))[0]

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in predict_classes]
        draw = ImageDraw.Draw(original_img)
        for box, cls, score, color in zip(predict_boxes, predict_classes, predict_scores, colors):
            if score < 0.5:
                continue
            x0, y0, x1, y1 = box
            # 绘制目标边界框
            draw.rectangle([x0, y0, x1, y1], width=6, outline=color)

        plt.imshow(original_img)
        plt.show()
        # 保存预测的图片结果
        original_img.save("dcp.jpg")


if __name__ == '__main__':
    main()