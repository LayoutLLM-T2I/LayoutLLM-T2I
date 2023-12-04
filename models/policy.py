from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch

from tools.aesthetic import AestheticMLP, normalized
from tools.metrics import compute_maximum_iou, compute_docsim
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self,
                 model_config = 'openai/clip-vit-base-patch32', 
                 add_linear=False,
                 in_dim=512, 
                 embedding_size=128, 
                 freeze_encoder=True) -> None:
        super().__init__()
        assert freeze_encoder
        if add_linear:
            self.embedding_size = embedding_size
            self.linear = nn.Linear(in_dim, embedding_size)
        else:
            self.linear = None
    

    def forward(self, emb_inp):
        sentence_embedding = emb_inp

        if self.linear:
            sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size

        return sentence_embedding


class Reward(nn.Module):
    def __init__(self, model_config, aesthetic_ckpt, args, device='cuda'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        self.processor = AutoProcessor.from_pretrained(model_config)
        self.model = CLIPModel.from_pretrained(model_config).to(device)
        self.model.eval()
        self.device = device
        self.args = args

        self.aesthetic_model = AestheticMLP(self.model.projection_dim)
        ckpt = torch.load(aesthetic_ckpt, map_location='cpu')
        self.aesthetic_model.load_state_dict(ckpt)
        self.aesthetic_model.eval()
        self.emb_labels()


    @torch.no_grad()
    def emb_labels(self, ):
        assert self.args.img_dir.split('/')[-1] == 'train2014'
        self.labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']

        self.label2index = {}
        for i, l in enumerate(self.labels):
            self.label2index[l] = i
        
        inputs = self.tokenizer(self.labels, padding=True, return_tensors="pt").to(self.device)
        labels_emb = self.model.get_text_features(**inputs)
        self.labels_emb = F.normalize(labels_emb, dim=-1) 

    def label_to_id(self, layouts):
        new_layouts = []
        for i, (boxes, labels) in enumerate(layouts):
            idx_lst = []
            for label in labels:
                idx = self.label2index[label]
                idx_lst.append(idx)
            new_layouts.append((np.array(boxes), np.array(idx_lst)))
        return new_layouts

    def nn_close_set(self, layouts):
        new_layouts = []
        for i, (boxes, labels) in enumerate(layouts):
            new_labels = []
            for label in labels:
                if label in self.labels:
                    new_labels.append(label)
                else:
                    l_inputs = self.tokenizer([label], padding=True, return_tensors="pt").to(self.device)
                    l_emb = self.model.get_text_features(**l_inputs)
                    l_emb = F.normalize(l_emb, dim=-1) 
                    sim = l_emb @ self.labels_emb.t() 
                    nearest_id = sim.flatten().argmax().item()
                    new_label = self.labels[nearest_id]
                    new_labels.append(new_label)
                    # print(label + '-->' + new_label)
            new_layouts.append((boxes, new_labels))
        
        return new_layouts


    @torch.no_grad()
    def forward(self, captions, imgs_pred, imgs_gt, layout_pred, layout_gt):
        # clip reward
        txt_inp = self.tokenizer(captions, padding=True, return_tensors="pt").to(self.device)
        txt_features = self.model.get_text_features(**txt_inp)
        imgs_pred = self.processor(images=imgs_pred, return_tensors="pt").to(self.device)
        imgs_pred_features = self.model.get_image_features(**imgs_pred)
        imgs_gt = self.processor(images=imgs_gt, return_tensors="pt").to(self.device)
        imgs_gt_features = self.model.get_image_features(**imgs_gt)
        txt_features = F.normalize(txt_features, dim=-1)
        imgs_pred_features = F.normalize(imgs_pred_features, dim=-1)
        imgs_gt_features = F.normalize(imgs_gt_features, dim=-1)
        sims_ti = (txt_features * imgs_pred_features).sum(dim=-1)
        sims_ii = (imgs_gt_features * imgs_pred_features).sum(dim=-1)
        clip_reward = sims_ti + sims_ii 

        # aesthetic reward
        im_emb_arr = normalized(imgs_pred_features.cpu().detach().numpy())
        aes_reward = self.aesthetic_model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
        aes_reward = aes_reward.flatten()

        # iou reward
        layout_pred_close = self.nn_close_set(layout_pred)
        layout_pred_id = self.label_to_id(layout_pred_close)
        layout_gt_id = self.label_to_id(layout_gt)
        miou = compute_maximum_iou(layout_gt_id, layout_pred_id)
        miou = torch.from_numpy(miou).to(self.device)

        # laysim reward
        laysim = compute_docsim(layout_gt_id, layout_pred_id)
        laysim = torch.from_numpy(laysim).to(self.device)
        
        reward = clip_reward + aes_reward * 0.1 + miou * 10 + laysim * 10
        return reward