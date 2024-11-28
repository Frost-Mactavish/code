import os
import json
import copy
from glob import glob
from typing import Optional

from PIL import Image
import xml.etree.ElementTree as ET

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

DIOR = {
    'joint': ("airplane", "baseballfield", "bridge", "groundtrackfield", "vehicle",
              "ship", "tenniscourt", "airport", "chimney", "dam",
              "basketballcourt", "Expressway-Service-area", "Expressway-toll-station", "golffield", "harbor",
              "overpass", "stadium", "storagetank", "trainstation", "windmill"),

    'base': ("airplane", "baseballfield", "bridge", "groundtrackfield", "vehicle",
             "ship", "tenniscourt", "airport", "chimney", "dam"),

    'inc': ("basketballcourt", "Expressway-Service-area", "Expressway-toll-station", "golffield", "harbor",
            "overpass", "stadium", "storagetank", "trainstation", "windmill")
}

DOTA = {
    'joint': ("plane", "baseball diamond", "bridge", "ground track field", "small vehicle",
              "large vehicle", "ship", "tennis court", "basketball court", "storage tank",
              "soccer ball field", "roundabout", "harbor", "swimming pool", "helicopter"),

    'base': ("plane", "baseball diamond", "bridge", "ground track field", "small vehicle",
             "large vehicle", "ship", "tennis court"),

    'inc': ("basketball court", "storage tank", "soccer ball field", "roundabout",
            "harbor", "swimming pool", "helicopter")
}


def get_class_dict(dataset_name: str, phase: str):
    '''
    map class string to number, given dataset name and training phase

    Args:
        dataset_name (str): name of dataset, i.e. 'DOTA' or 'DIOR'
        phase (str): phase of training, i.e. 'joint', 'base', 'inc'

    Returns:
        dict mapping class string to number, as specified above
    '''
    assert dataset_name in ['DOTA', 'DIOR']
    assert phase in ['joint', 'base', 'inc']
    dataset = eval(dataset_name)

    return {k: {str: (i+1) for i, str in enumerate(dataset[k])} for k in dataset.keys()}[phase]


DIOR_reverse = ("basketballcourt", "Expressway-Service-area", "Expressway-toll-station", "golffield", "harbor",
        "overpass", "stadium", "storagetank", "trainstation", "windmill", "airplane", "baseballfield",
        "bridge", "groundtrackfield", "vehicle", "ship", "tenniscourt", "airport", "chimney", "dam")


class DIORIncDataset(Dataset):
    '''
    Args:
        root (str): root directory of dataset -> path to fetch data
        transform (object): organize data transform pipeline, likes of torchvision.transforms.Compose
        dataset (str): name of dataset -> get_class_dict()
        mode (str): train or test -> what data to fetch
        phase (str): incremental phase -> whether or not filter classes
    '''
    def __init__(self, root, transform=None, mode='train', phase='base'):
        assert mode in ['train', 'test']
        assert phase in ['joint', 'base', 'inc']

        self.img_dir = os.path.join(root, 'Images')
        self.xml_dir = os.path.join(root, 'Annotations/HBB')
        self.transform = transform

        txt_path = os.path.join(root, 'ImageSets', mode + '.txt')

        self.xml_list = []
        with open (txt_path, 'r') as f:
            for line in f:
                xml = os.path.join(self.xml_dir, line.strip('\n') + '.xml')
                self.xml_list.append(xml)

                # if len(self.xml_list) == 100:       # 小规模数据加载，快速验证训练流程
                #     break

        self.class_dict = get_class_dict('DIOR', phase)
        # self.class_dict = {value: (i+1) for i, value in enumerate(DIOR_reverse)}
        if phase in ['base', 'inc']:
            self.filter_classes()

    def filter_classes(self):
        xml_list = copy.deepcopy(self.xml_list)
        for xml_path in xml_list:
            with open(xml_path, 'r') as f:
                xml_str = f.read()
            xml = ET.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)['annotation']

            count = 0
            for obj in data['object']:
                if obj['name'] in self.class_dict.keys():
                    count += 1
            if count == 0:
                self.xml_list.remove(xml_path)

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
            if obj['name'] in self.class_dict.keys():
                obj_class = self.class_dict[obj['name']]

                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                if xmax <= xmin or ymax <= ymin:
                    continue

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


class CalcDataset(Dataset):
    '''
    calc mean, std of dataset

    Args:
        root (str): root directory of dataset
        normalize (bool, optional): whether to rescale data to [0, 1]
    '''
    def __init__(self, root: str, normalize: Optional[bool] = True):
        self.transform = transforms.ToTensor() if normalize else None
        self.img_list = glob(f'{root}/*')

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.img_list)

    def calc(self, batch_size=512):
        dataloader = DataLoader(self, batch_size, num_workers=8)

        sum, squared_sum, batch_num = 0, 0, 0
        for img in dataloader:
            sum += torch.mean(img, dim=[0, 2, 3])
            squared_sum += torch.mean(img ** 2, dim=[0, 2, 3])
            batch_num += 1
        mean = (sum / batch_num).tolist()
        std = ((squared_sum / batch_num - mean ** 2) ** 0.5).tolist()

        return mean, std


if __name__ == '__main__':
    root = '/home/freddy/code/dataset/DIOR/Images'
    dataset = CalcDataset(root)
    mean, std = dataset.calc()

    print(f'mean:{mean}, std:{std}')
