import os
import json
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageColor, ImageDraw

import torch
from torchvision import transforms

from dataset import get_class_dict
from detection.model import load_weights
from utils.train_eval_utils import STANDARD_COLORS


def main(args):
    with open('config.json') as f:
        config = json.load(f)[args.dataset]

    root = config['root']
    mean, std = config['mean'], config['std']
    save_dir = config['save_dir']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weight_path = os.path.join(save_dir, args.filename)
    assert os.path.exists(weight_path)
    model, phase = load_weights(weight_path)
    model.to(device)

    class_dict = get_class_dict(args.dataset, phase)
    category_index = {str(v): str(k) for k, v in class_dict.items()}

    img_list = os.listdir(root)
    img_path = random.sample(img_list, 1)[0]
    original_img = Image.open(img_path)

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        predictions = model(img.to(device))[0]

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_classes = category_index[predict_classes]
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("No object detected.")

        colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in predict_classes]
        draw = ImageDraw.Draw(original_img)
        for box, cls, score, color in zip(predict_boxes, predict_classes, predict_scores, colors):
            if score < 0.5:
                continue
            x0, y0, x1, y1 = box
            draw.rectangle([x0, y0, x1, y1], width=6, outline=color)

        plt.imshow(original_img)
        plt.show()
        original_img.save("detection_result.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='DIOR', help='dataset name')
    parser.add_argument('--filename', default='DIOR_50_joint_72.68.pth',
                        help='filename of weight')
    args = parser.parse_args()
    print(args)

    main(args)