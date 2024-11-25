import os
import glob
import json
from PIL import Image
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset


class DIORDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.img_dir = os.path.join(root, 'Images')
        self.xml_dir = os.path.join(root, 'Annotations/HBB')
        self.transform = transform

        txt_path = os.path.join(root, 'ImageSets', mode + '.txt')

        self.xml_list = []
        with open (txt_path, 'r') as f:
            for line in f:
                xml = os.path.join(self.xml_dir, line.strip('\n') + '.xml')
                self.xml_list.append(xml)

                # if len(self.xml_list) == 100:     # 小规模数据加载，快速验证训练流程
                #     break

        with open('classes.json', 'r') as f:
            self.class_dict = json.load(f)["DIOR"]

    def __getitem__(self, idx):
        xml_path = self.xml_list[idx]
        with open (xml_path, 'r') as f:
            xml_str = f.read()
        xml = ET.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        img_path = os.path.join(self.img_dir, data['filename'])
        img = Image.open(img_path)

        boxes = []
        labels = []
        iscrowd = []
        for obj in data['object']:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            if xmax <= xmin or ymax <= ymin:
                continue

            obj_class = self.class_dict[obj['name']]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj_class)
            iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['filename'] = data['filename']

        if self.transform is not None:
            img, target['boxes'] = self.transform(img, target['boxes'])

        return img, target

    def __len__(self):
        return len(self.xml_list)

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class DOTADataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        if mode not in ['train', 'val']:
            raise ValueError("Param mode value wrong, should be 'train' or 'val'!")

        self.img_dir = os.path.join(root, mode, 'images')
        self.label_dir = os.path.join(root, mode, 'labelTxt-v1.0')
        self.transform = transform
        self.file_list = [os.path.basename(file_name).split('.')[0] for file_name in glob.glob(self.img_dir + "/*.png")]

        with open('classes.json', 'r') as f:
            self.class_dict = json.load(f)["DOTA"]

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, file_name + '.png')
        label_path = os.path.join(self.label_dir, file_name + '.txt')

        img = Image.open(img_path)

        boxes = []
        labels = []
        iscrowd = []
        with open(label_path, 'r') as txt:
            for line in txt:
                line = line.split[' ']
                [xmin, ymin], [xmax, ymax], [label, diffcult] = line[:2], line[4:6], line[-2:]

                if xmax <= xmin or ymax <= ymin:
                    continue

            object_class = self.class_dict[obj['name']]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(object_class)
            iscrowd.append(int(diffcult))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['filename'] = file_name

        if self.transform is not None:
            img, target['boxes'] = self.transform(img, target['boxes'])

        return img, target

    def __len__(self):
        return len(self.xml_list)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class CalcDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.img_list = glob.glob(img_dir + '/*.jpg')

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from torchvision import transforms

    # 计算数据集图像的mean,std
    transform = transforms.ToTensor()
    root = '/home/freddy/code/dataset/DIOR/test_ridcp'
    dataset = CalcDataset(root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=512)

    sum, squared_sum, batch_num = 0, 0, 0
    for img in dataloader:
        sum += torch.mean(img, dim=[0, 2, 3])
        squared_sum += torch.mean(img ** 2, dim=[0, 2, 3])
        batch_num += 1
    mean = (sum / batch_num)
    std = ((squared_sum / batch_num - mean ** 2) ** 0.5)
    print('mean:{},std:{}'.format(mean, std))
