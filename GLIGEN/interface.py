import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import argparse
import json
import warnings
from typing import Union

import sng_parser
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from torch import FloatTensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch
from ldm.util import instantiate_from_config
from trainer import read_official_ckpt, batch_to_device
from inpaint_mask_func import draw_masks_from_boxes
import numpy as np
import clip
from scipy.io import loadmat
from functools import partial
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import sng_parser
from pprint import pprint


def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling.
    type should be a list containing three values which sum should be 1

    It means the percentage of three stages:
    alpha=1 stage
    linear deacy stage
    alpha=0 stage.

    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.
    """
    if type == None:
        type = [1, 0, 0]

    assert len(type) == 3
    assert type[0] + type[1] + type[2] == 1

    stage0_length = int(type[0] * length)
    stage1_length = int(type[1] * length)
    stage2_length = length - stage0_length - stage1_length

    if stage1_length != 0:
        decay_alphas = np.arange(start=0, stop=1, step=1 / stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []

    alphas = [1] * stage0_length + decay_alphas + [0] * stage2_length

    assert len(alphas) == length

    return alphas


def load_ckpt(ckpt_path, device="cuda"):
    saved_ckpt = torch.load(ckpt_path, map_location='cpu')
    config = saved_ckpt["config_dict"]["_content"]
    # config['model']['target'] = 'ldm.modules.diffusionmodules.gligen_combine_layout.GligenCombineLayout'
    # print(config)

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    # sub_state_dict = read_pre_trained_ckpt(saved_ckpt)
    model.load_state_dict(saved_ckpt['model'], strict=False)
    autoencoder.load_state_dict(saved_ckpt["autoencoder"])
    text_encoder.load_state_dict(saved_ckpt["text_encoder"])
    diffusion.load_state_dict(saved_ckpt["diffusion"])

    all_models = [model, autoencoder, text_encoder, diffusion]
    for m in all_models:
        if 'device' in vars(m):
            m.device = device

    return model, autoencoder, text_encoder, diffusion, config


def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.
    this function will return the CLIP feature (without normalziation)
    """
    return x @ torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, device, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].to(device)  # we use our own preprocessing without center_crop
        inputs['input_ids'] = torch.tensor([[0, 1, 2, 3]]).to(device)  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds
        if which_layer_image == 'after_reproject':
            feature = project(feature, torch.load('projection_matrix').to(device).T).squeeze(0)
            feature = (feature / feature.norm()) * 28.7
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input, return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['pixel_values'] = torch.ones(1, 3, 224, 224).to(device)  # placeholder
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1, max_objs)
    if has_mask == None:
        return mask

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0, idx] = value
        return mask


@torch.no_grad()
def prepare_batch(meta, model, processor, batch=1, max_objs=30, device=None):
    phrases, images = meta.get("phrases"), meta.get("images")
    images = [None] * len(phrases) if images == None else images
    phrases = [None] * len(images) if phrases == None else phrases

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)

    text_features = []
    image_features = []
    for phrase, image in zip(phrases, images):
        text_features.append(get_clip_feature(model, processor, phrase, device, is_image=False))
        image_features.append(get_clip_feature(model, processor, image, device, is_image=True))

    for idx, (box, text_feature, image_feature) in enumerate(zip(meta['locations'], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1

    out = {
        "boxes": boxes.unsqueeze(0).repeat(batch, 1, 1),
        "masks": masks.unsqueeze(0).repeat(batch, 1),
        "text_masks": text_masks.unsqueeze(0).repeat(batch, 1) * complete_mask(meta.get("text_mask"), max_objs),
        "image_masks": image_masks.unsqueeze(0).repeat(batch, 1) * complete_mask(meta.get("image_mask"), max_objs),
        "text_embeddings": text_embeddings.unsqueeze(0).repeat(batch, 1, 1),
        "image_embeddings": image_embeddings.unsqueeze(0).repeat(batch, 1, 1)
    }

    return batch_to_device(out, device)


def crop_and_resize(image):
    crop_size = min(image.size)
    image = F.center_crop(image, crop_size)
    image = image.resize((512, 512))
    return image


def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)

    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
                        np.tile(colors[label],
                                (labelmap.shape[0], labelmap.shape[1], 1))

    return labelmap_rgb


@torch.no_grad()
def prepare_relation_phrases(prompt, batch_size=1, max_relas=5, text_encoder=None, device=None):
    """
    parse the relation triplet in the prompt, and get the relation triplet embeddings
    """
    graph = sng_parser.parse(prompt)
    relations = ["PAD"]
    entities = graph["entities"]
    if "relations" in graph:
        for r in graph["relations"]:
            obj = entities[r["object"]]["lemma_head"]
            subj = entities[r["subject"]]["lemma_head"]
            rela = r["relation"]
            relations.append(' '.join([subj, rela, obj]))

    if "relations" in graph and len(graph["relations"]) > 0:
        for r in graph["relations"]:
            obj = entities[r["object"]]["lemma_head"]
            subj = entities[r["subject"]]["lemma_head"]
            rela = r["relation"]
            relations.append(' '.join([subj, rela, obj]))

        if len(relations) > max_relas:
            relations = relations[:max_relas]
            _, relation_embeddings = text_encoder.encode(relations[:max_relas], return_pooler_output=True)
        else:
            relation_embeddings = torch.zeros(max_relas, 768)
            _, _tmp = text_encoder.encode(relations, return_pooler_output=True)
            relation_embeddings[:len(relations), :] = _tmp
    else:
        relation_embeddings = torch.zeros(max_relas, 768)

    return relation_embeddings.unsqueeze(0).repeat(batch_size, 1, 1).to(device)


@torch.no_grad()
def get_clip_score(pred_image, glod_image, caption, model, processor, w=2.5):
    inputs = processor(images=[pred_image], return_tensors="pt", padding=True)
    inputs['pixel_values'] = inputs['pixel_values'].cuda()  # we use our own preprocessing without center_crop
    inputs['input_ids'] = torch.tensor([[0, 1, 2, 3]]).cuda()  # placeholder
    outputs = model(**inputs)
    feature = outputs.image_embeds
    feature = project(feature, torch.load('projection_matrix').cuda().T).squeeze(0)
    feature = (feature / feature.norm()) * 28.7
    pred_feature = feature.unsqueeze(0).cpu().numpy()

    inputs = processor(images=[glod_image], return_tensors="pt", padding=True)
    inputs['pixel_values'] = inputs['pixel_values'].cuda()  # we use our own preprocessing without center_crop
    inputs['input_ids'] = torch.tensor([[0, 1, 2, 3]]).cuda()  # placeholder
    outputs = model(**inputs)
    feature = outputs.image_embeds
    feature = project(feature, torch.load('projection_matrix').cuda().T).squeeze(0)
    feature = (feature / feature.norm()) * 28.7
    gold_feature = feature.unsqueeze(0).cpu().numpy()

    inputs = processor(text=caption, return_tensors="pt", padding=True)
    inputs['input_ids'] = inputs['input_ids'].cuda()
    inputs['pixel_values'] = torch.ones(1, 3, 224, 224).cuda()  # placeholder
    inputs['attention_mask'] = inputs['attention_mask'].cuda()
    outputs = model(**inputs)
    # if which_layer_text == 'before':
    text_feature = outputs.text_model_output.pooler_output.cpu().numpy()

    pred_images = pred_feature / np.sqrt(np.sum(pred_feature ** 2, axis=1, keepdims=True))
    gold_images = gold_feature / np.sqrt(np.sum(gold_feature ** 2, axis=1, keepdims=True))
    candidates = text_feature / np.sqrt(np.sum(text_feature ** 2, axis=1, keepdims=True))

    t_2_i = w * np.clip(np.sum(pred_images * candidates, axis=1), 0, None)
    i_2_i = w * np.clip(np.sum(gold_images * pred_images, axis=1), 0, None)
    return np.mean(t_2_i), np.mean(i_2_i)


@torch.no_grad()
def run_one_image(all_models, args, meta, starting_noise=None, clip_model=None, clip_processor=None, device=None):
    # - - - - - prepare models - - - - - #
    model, autoencoder, text_encoder, diffusion, config = all_models

    # - - - - - update config from args - - - - - #
    config.update(args)
    config = OmegaConf.create(config)

    # - - - - - prepare batch - - - - - #
    batch = prepare_batch(meta, clip_model, clip_processor, config.batch_size, device=device)

    context = text_encoder.encode([meta["prompt"]] * config.batch_size)
    relations = prepare_relation_phrases(meta["prompt"], config.batch_size, config.max_relations, text_encoder, device=device)

    uc = text_encoder.encode(config.batch_size * [""])
    # if args.negative_prompt is not None:
    #     uc = text_encoder.encode(config.batch_size * [args.negative_prompt])

    # - - - - - sampler - - - - - #
    alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
    if config.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)
        steps = 250
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)
        steps = 50

        # - - - - - inpainting related - - - - - #
    inpainting_mask = z0 = None  # used for replacing known region in diffusion process
    inpainting_extra_input = None  # used as model input

    # - - - - - input for gligen - - - - - #
    grounding_input = model.grounding_tokenizer_input.prepare(batch, text_encoder)
    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])
    grounding_extra_input = None
    if grounding_downsampler_input != None:
        grounding_extra_input = grounding_downsampler_input.prepare(batch)

    input = dict(
        x=starting_noise,
        timesteps=None,
        context=context,
        relations=relations,
        grounding_input=grounding_input,
        inpainting_extra_input=inpainting_extra_input,
        grounding_extra_input=grounding_extra_input,
    )

    # - - - - - start sampling - - - - - #
    shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)
    samples_fake = sampler.sample(S=steps, shape=shape, input=input, uc=uc, guidance_scale=config.guidance_scale,
                                  mask=inpainting_mask, x0=z0)
    samples_fake = autoencoder.decode(samples_fake)
    imgs_gen = []
    for sample in samples_fake:
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1, 2, 0) * 255
        sample = Image.fromarray(sample.astype(np.uint8))
        imgs_gen.append(sample)

    return imgs_gen


def load_json(path):
    with open(path, 'r') as f:
        x = json.load(f)
    return x


def load_all_models(ckpt, device):
    # - - - - - prepare models - - - - - #
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(ckpt, device)
    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    all_models = (model, autoencoder, text_encoder, diffusion, config)

    return all_models


def generate_one_image(all_models, caption, label, bbox, clip_model=None, clip_processor=None, device=None):
    args = dict(
        batch_size=1,
        no_plms=False,
        guidance_scale=7.5
    )

    bbox = [convert_xywh_to_ltrb(b) for b in bbox]
    meta = dict(
        prompt=caption,
        phrases=label,
        locations=bbox,
        alpha_type=[0.3, 0.0, 0.7],
    )

    starting_noise = torch.randn(args['batch_size'], 4, 64, 64).to(device)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    imgs_gen = run_one_image(all_models, args, meta, starting_noise, clip_model, clip_processor, device=device)
    return imgs_gen


def convert_xcycwh_to_ltrb(bbox: Union[np.ndarray, FloatTensor]):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def convert_xywh_to_ltrb(bbox: Union[np.ndarray, FloatTensor]):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


def draw_box(sample, locations, phrases):
    draw = ImageDraw.Draw(sample)
    w, h = sample.size
    for i in range(len(locations)):
        x0, y0, x1, y1 = locations[i]
        x0, y0, x1, y1 = x0 * w, y0 * h, x1 * w, y1 * h  # (x0,y0) upper left，（x1,y1）low right
        draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 255), width=4)
        font = ImageFont.truetype("/storage/lgqu/my_fonts/arial.ttf", size=20)
        draw.text((x0, y0-18), phrases[i], (255, 0, 0), font=font)

@torch.no_grad()
def prepare_batch_multiple(meta, model, processor, batch=1, max_objs=30, device=None):
    phrases_batch, images_batch = meta.get("phrases"), meta.get("images")
    assert batch == len(phrases_batch)
    locations_batch = meta['locations']

    out_batch = []
    for i_data, phrases in enumerate(phrases_batch):
        images = None
        # phrases, images = meta.get("phrases"), meta.get("images")
        images = [None] * len(phrases) if images == None else images
        phrases = [None] * len(images) if phrases == None else phrases

        boxes = torch.zeros(max_objs, 4)
        masks = torch.zeros(max_objs)
        text_masks = torch.zeros(max_objs)
        image_masks = torch.zeros(max_objs)
        text_embeddings = torch.zeros(max_objs, 768)
        image_embeddings = torch.zeros(max_objs, 768)

        text_features = []
        image_features = []
        for phrase, image in zip(phrases, images):
            text_features.append(get_clip_feature(model, processor, phrase, device, is_image=False))
            image_features.append(get_clip_feature(model, processor, image, device, is_image=True))

        for idx, (box, text_feature, image_feature) in enumerate(zip(locations_batch[i_data], text_features, image_features)):
            boxes[idx] = torch.tensor(box)
            masks[idx] = 1
            if text_feature is not None:
                text_embeddings[idx] = text_feature
                text_masks[idx] = 1
            if image_feature is not None:
                image_embeddings[idx] = image_feature
                image_masks[idx] = 1

        out = {
            "boxes": boxes.unsqueeze(0),
            "masks": masks.unsqueeze(0),
            "text_masks": text_masks.unsqueeze(0) * complete_mask(meta.get("text_mask"), max_objs),
            "image_masks": image_masks.unsqueeze(0) * complete_mask(meta.get("image_mask"), max_objs),
            "text_embeddings": text_embeddings.unsqueeze(0),
            "image_embeddings": image_embeddings.unsqueeze(0)
        }
        out_batch.append(out)
    
    out_new = {}
    for k in out_batch[0].keys():
        out_new[k] = torch.cat([o[k] for o in out_batch], dim=0)


    return batch_to_device(out_new, device)


@torch.no_grad()
def run_batch_images(all_models, args, meta, starting_noise=None, clip_model=None, clip_processor=None, device=None):
    # - - - - - prepare models - - - - - #
    model, autoencoder, text_encoder, diffusion, config = all_models

    # - - - - - update config from args - - - - - #
    config.update(args)
    config = OmegaConf.create(config)

    # - - - - - prepare batch - - - - - #
    batch = prepare_batch_multiple(meta, clip_model, clip_processor, config.batch_size, device=device)

    context = text_encoder.encode(meta["prompts"])
    relations = [prepare_relation_phrases(p, 1, config.max_relations, text_encoder, device=device)
                        for p in meta["prompts"]]
    relations = torch.cat(relations, dim=0)
    

    uc = text_encoder.encode(config.batch_size * [""])
    # if args.negative_prompt is not None:
    #     uc = text_encoder.encode(config.batch_size * [args.negative_prompt])

    # - - - - - sampler - - - - - #
    alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
    if config.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)
        steps = 250
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)
        steps = 50

        # - - - - - inpainting related - - - - - #
    inpainting_mask = z0 = None  # used for replacing known region in diffusion process
    inpainting_extra_input = None  # used as model input

    # - - - - - input for gligen - - - - - #
    grounding_input = model.grounding_tokenizer_input.prepare(batch, text_encoder)
    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])
    grounding_extra_input = None
    if grounding_downsampler_input != None:
        grounding_extra_input = grounding_downsampler_input.prepare(batch)

    # print(config)
    # print(starting_noise.shape, context.shape, relations.shape)
    # print(uc.shape)
    input = dict(
        x=starting_noise,
        timesteps=None,
        context=context,
        relations=relations,
        grounding_input=grounding_input,
        inpainting_extra_input=inpainting_extra_input,
        grounding_extra_input=grounding_extra_input,
    )

    # - - - - - start sampling - - - - - #
    shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)
    samples_fake = sampler.sample(S=steps, shape=shape, input=input, uc=uc, guidance_scale=config.guidance_scale,
                                  mask=inpainting_mask, x0=z0)
    samples_fake = autoencoder.decode(samples_fake)
    imgs_gen = []
    for sample in samples_fake:
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1, 2, 0) * 255
        sample = Image.fromarray(sample.astype(np.uint8))
        imgs_gen.append(sample)

    return imgs_gen

def generate_batch_images(all_models, captions, labels, bboxes, clip_model=None, clip_processor=None, device=None):
    bs = len(captions)
    args = dict(
        batch_size=bs,
        no_plms=False,
        guidance_scale=7.5
    )

    meta = dict(
        prompts=captions,
        phrases=labels,
        locations=bboxes,
        alpha_type=[0.3, 0.0, 0.7],
    )

    starting_noise = torch.randn(bs, 4, 64, 64).to(device)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    imgs_gen = run_batch_images(all_models, args, meta, starting_noise, clip_model, clip_processor, device=device)
    return imgs_gen