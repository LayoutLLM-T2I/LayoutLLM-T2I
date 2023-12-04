import json, os, random, math
import pickle
from collections import defaultdict
from copy import deepcopy

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .base_dataset import BaseDataset, check_filenames_in_zipdata, to_valid_bbox
from io import BytesIO
from pycocotools.coco import COCO
from tqdm import tqdm


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def clean_kps(kps):
    assert len(kps) == 51
    kps = list(chunks(kps, 3))
    out = []
    for idx, kp in enumerate(kps):
        name = "kp" + str(idx).zfill(2)
        loc = [kp[0], kp[1]]
        valid = True if kp[2] == 2 else False
        if not valid:
            loc = [0, 0]
        out.append({"name": name, "loc": loc, "valid": valid})
    return out


def norm_kps(kps, image_size):
    for kp in kps:
        if kp["valid"]:
            kp_x, kp_y = kp["loc"]
            kp["loc"] = [kp_x / image_size, kp_y / image_size]
    return kps


def norm_bbox(bbox, image_size):
    x_0 = bbox[0] / image_size
    y_0 = bbox[1] / image_size
    x_1 = bbox[2] / image_size
    y_1 = bbox[3] / image_size
    return x_0, y_0, x_1, y_1


def clean_annotations(annotations):
    for anno in annotations:
        anno.pop("segmentation", None)
        anno.pop("area", None)
        anno.pop("iscrowd", None)
        anno.pop("id", None)


def check_all_have_same_images(instances_data, caption_data):
    if caption_data is not None:
        assert instances_data["images"] == caption_data["images"]


class LayoutDataset(BaseDataset):
    def __init__(self,
                 image_root,
                 layout_json_path=None,
                 caption_json_path=None,
                 prob_real_caption=0,
                 image_size=512,
                 max_images=None,
                 min_box_size=0.0,
                 max_persons_per_image=8,
                 random_crop=False,
                 random_flip=True,
                 version="openai/clip-vit-large-patch14",
                 preprocess_data_dir=None
                 ):
        super().__init__(random_crop, random_flip, image_size)

        self.image_root = image_root
        self.layout_json_path = layout_json_path
        self.caption_json_path = caption_json_path
        self.prob_real_caption = prob_real_caption
        self.max_images = max_images
        self.min_box_size = min_box_size
        self.max_persons_per_image = max_persons_per_image

        self.version = version

        # self.box_label_model = CLIPModel.from_pretrained(self.version).cuda()
        # self.box_label_processor = CLIPProcessor.from_pretrained(self.version)
        # self.freeze()
        # free CLIP model
        # box_label_model = box_label_model.eval()
        # for param in box_label_model.parameters():
        #     param.requires_grad = False

        coco_instance = COCO(self.layout_json_path)
        coco_caption = COCO(self.caption_json_path)

        img_ids = sorted(coco_instance.getImgIds())
        # img_ids = sorted(coco_instance.getImgIds())
        self.data_list = []  # no numeral,spatial, semantic
        for img_id in tqdm(img_ids):
            ann_img = coco_instance.loadImgs(img_id)
            W = float(ann_img[0]["width"])
            H = float(ann_img[0]["height"])
            name = ann_img[0]["file_name"]

            elements_instance = coco_instance.loadAnns(coco_instance.getAnnIds(imgIds=[img_id]))
            elements_cap = coco_caption.loadAnns(coco_caption.getAnnIds(imgIds=[img_id]))
            caption = elements_cap[0]['caption']

            N = len(elements_instance)
            if N == 0 or self.max_persons_per_image < N:
                continue

            boxes = []
            labels = []
            # labels_emb = []

            for element in elements_instance:
                # bbox
                boxes.append(element["bbox"])

                # label
                l = coco_instance.cats[element["category_id"]]["name"]
                labels.append(l)
                # labels_emb.append(get_text_embedding(l))
            data = {
                "image_id": img_id,
                "name": name,
                "width": W,
                "height": H,
                "boxes": boxes,
                "caption": caption,
                "labels": labels,
                # "text_embeddings": labels_emb
            }
            self.data_list.append(data)
        #
        # self.preprocess_data_path = os.path.join(preprocess_data_dir, "data.pt")
        # if os.path.exists(self.preprocess_data_path):
        #     with open(self.preprocess_data_path, 'rb') as f:
        #         self.data_list = pickle.load(f)
        # else:
        #     self.data_list = self.preprocess()

    def preprocess(self):
        from pycocotools.coco import COCO
        from tqdm import tqdm
        box_label_model = CLIPModel.from_pretrained(self.version).cuda()
        box_label_processor = CLIPProcessor.from_pretrained(self.version)

        # free CLIP model
        box_label_model = box_label_model.eval()
        for param in box_label_model.parameters():
            param.requires_grad = False

        coco_instance = COCO(self.layout_json_path)
        coco_caption = COCO(self.caption_json_path)

        img_ids = sorted(coco_instance.getImgIds())
        # img_ids = sorted(coco_instance.getImgIds())
        data_list = []  # no numeral,spatial, semantic
        for img_id in tqdm(img_ids):
            ann_img = coco_instance.loadImgs(img_id)
            W = float(ann_img[0]["width"])
            H = float(ann_img[0]["height"])
            name = ann_img[0]["file_name"]

            elements_instance = coco_instance.loadAnns(coco_instance.getAnnIds(imgIds=[img_id]))
            elements_cap = coco_caption.loadAnns(coco_caption.getAnnIds(imgIds=[img_id]))
            caption = elements_cap[0]['caption']

            N = len(elements_instance)
            if N == 0 or self.max_persons_per_image < N:
                continue

            boxes = []
            labels = []
            labels_emb = []

            @torch.no_grad()
            def get_text_embedding(t):
                with torch.no_grad():
                    inputs = box_label_processor(text=t, return_tensors="pt", padding=True)
                    inputs['input_ids'] = inputs['input_ids'].cuda()
                    inputs['pixel_values'] = torch.ones(1, 3, 224, 224).cuda()  # placeholder
                    inputs['attention_mask'] = inputs['attention_mask'].cuda()
                    outputs = box_label_model(**inputs)
                    feature = outputs.text_model_output.pooler_output
                    return feature

            for element in elements_instance:
                # bbox
                boxes.append(element["bbox"])

                # label
                l = coco_instance.cats[element["category_id"]]["name"]
                labels.append(l)
                labels_emb.append(get_text_embedding(l))
            data = {
                "image_id": img_id,
                "name": name,
                "width": W,
                "height": H,
                "boxes": boxes,
                "caption": caption,
                "labels": labels,
                "text_embeddings": labels_emb
            }
            data_list.append(data)
        if os.path.exists(self.preprocess_data_path):
            with open(os.path.join(self.preprocess_data_path), 'rb') as f:
                pickle.dump(data_list, f)
        return data_list

    def freeze(self):
        self.box_label_model = self.box_label_model.eval()
        for param in self.box_label_model.parameters():
            param.requires_grad = False

    def select_objects(self, annotations):
        for object_anno in annotations:
            image_id = object_anno['image_id']
            self.image_id_to_objects[image_id].append(object_anno)

    def select_captions(self, annotations):
        for caption_data in annotations:
            image_id = caption_data['image_id']
            self.image_id_to_captions[image_id].append(caption_data)

    def total_images(self):
        return len(self)

    def __getitem__(self, index):
        if self.max_persons_per_image > 99:
            assert False, "Are you sure setting such large number of boxes?"

        out = {}

        image_id = self.data_list[index]["image_id"]
        out['id'] = image_id
        # image_id = 18150 #180560
        # Image
        filename = self.data_list[index]["name"]
        image = Image.open(os.path.join(self.image_root, filename)).convert('RGB')
        image_tensor, trans_info = self.transform_image(image)
        out["image"] = image_tensor

        areas = []
        all_bbox = []
        all_labels = []
        # all_labels_emb = []
        for b, l in zip(deepcopy(self.data_list[index]["boxes"]), deepcopy(self.data_list[index]["labels"])):

            x, y, w, h = b
            valid, (x0, y0, x1, y1) = to_valid_bbox(x, y, w, h, trans_info, self.image_size, self.min_box_size)

            if valid:
                areas.append((x1 - x0) * (y1 - y0))
                all_bbox.append(norm_bbox([x0, y0, x1, y1], self.image_size))
                all_labels.append(l)
                # all_labels_emb.append(l_emb)

        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[0:self.max_persons_per_image]

        bboxes = torch.zeros(self.max_persons_per_image, 4)
        masks = torch.zeros(self.max_persons_per_image).to(torch.int)
        # text_embedding = torch.zeros(self.max_persons_per_image, 768)
        labels = ["PAD"] * self.max_persons_per_image

        # print(labels)
        # print(self.data_list[index]["labels"])
        # print(all_labels)

        j = 0
        for idx in wanted_idxs:
            bboxes[j] = torch.tensor(all_bbox[idx])
            masks[j] = 1
            labels[j] = all_labels[idx]
            j += 1
        # print(labels)

        out["caption"] = self.data_list[index]["caption"]
        out["labels"] = '|'.join(labels)
        out["masks"] = masks
        out["boxes"] = bboxes

        return out

    def __len__(self):
        if self.max_images is None:
            return len(self.data_list)
        return min(len(self.data_list), self.max_images)

