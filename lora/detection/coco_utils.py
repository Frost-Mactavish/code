import torch
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, Subset

from pycocotools.coco import COCO


def get_coco_api_from_dataset(dataset):
    if isinstance(dataset, CocoDetection):
        return dataset.coco
    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    return convert_to_coco_api(dataset)


def convert_to_coco_api(ds):
    current_time = time.time()
    dataloader = DataLoader(dataset=ds, batch_size=16, num_workers=8,
                            pin_memory=True, collate_fn=ds.collate_fn)

    coco_ds = COCO()
    ann_id = 0
    categories = set()
    dataset = {'images': [], 'categories': [], 'annotations': []}
    for [imgs, targets] in dataloader:
        for idx in range(len(imgs)):
            img, target = imgs[idx], targets[idx]

            image_id = target["image_id"].item()

            img_dict = {}
            img_dict['id'] = image_id
            img_dict['height'] = img.shape[-2]
            img_dict['width'] = img.shape[-1]
            dataset['images'].append(img_dict)

            bboxes = target["boxes"]
            bboxes[:, 2:] -= bboxes[:, :2]
            bboxes = bboxes.tolist()

            labels = target['labels'].tolist()
            areas = target['area'].tolist()
            iscrowd = target['iscrowd'].tolist()

            if 'masks' in target:
                masks = target['masks']
                # make masks Fortran contiguous for coco_mask
                masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)

            if 'keypoints' in target:
                keypoints = target['keypoints']
                keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()

            for i in range(len(bboxes)):
                ann = {}
                ann['image_id'] = image_id
                ann['bbox'] = bboxes[i]
                ann['category_id'] = labels[i]
                ann['area'] = areas[i]
                ann['iscrowd'] = iscrowd[i]
                ann['id'] = ann_id

                categories.add(labels[i])

                if 'masks' in target:
                    ann["segmentation"] = coco_mask.encode(masks[i].numpy())

                if 'keypoints' in target:
                    ann['keypoints'] = keypoints[i]
                    ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])

                dataset['annotations'].append(ann)
                ann_id += 1

    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()

    return coco_ds