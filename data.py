import os
import numpy as np
import PIL
from PIL import Image
import cv2
import torch
import json
import torch.utils.data
import string 
from basicsr.utils import img2tensor

import utils

class COCO2014(torch.utils.data.Dataset): 
    def __init__(self, annotations, features_cap, img_dir):
        super(COCO2014, self).__init__()
        assert len(annotations) == len(features_cap)
        self.annotations = annotations
        self.features_cap = features_cap
        self.img_dir = img_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations[index]['name']
        img = Image.open(os.path.join(self.img_dir, img_name))
        img = img.convert('RGB')
        img = img.resize((512, 512))

        return self.annotations[index], self.features_cap[index], index, img


def collate_fn(data):
    anno, feat_cap, idx, img = zip(*data)
    feat_cap = torch.stack(feat_cap)
    return anno, feat_cap, idx, img


def load_data(args, reward):
    tokenizer, model, device = reward.tokenizer, reward.model, reward.device
    train_data = utils.load_json(os.path.join(args.sampled_data_dir, f'train2014_train_{args.train_number}.json'))
    train_ids, train_examples = train_data['id'], train_data['data']
    cand_data = utils.load_json(os.path.join(args.sampled_data_dir, f'train2014_candidate_{args.cand_number}.json'))
    cand_ids, cand_examples = cand_data['id'], cand_data['data']

    caps_train_data = [d['captions'] for d in train_examples]
    caps_cand_data = [d['captions'] for d in cand_examples]

    with torch.no_grad():
        inputs = tokenizer(caps_train_data, padding=True, return_tensors="pt").to(device)
        feats_caps_train = model.get_text_features(**inputs).cpu()
        inputs = tokenizer(caps_cand_data, padding=True, return_tensors="pt").to(device)
        feats_caps_cand = model.get_text_features(**inputs).cpu()

    return train_examples, cand_examples, train_ids, cand_ids, feats_caps_train, feats_caps_cand