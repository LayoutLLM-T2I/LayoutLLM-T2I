import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import argparse
import json
import warnings
from typing import Union
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer

import sng_parser
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from torch import FloatTensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
sys.path.append("./GLIGEN")
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
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
import time
from torch import autocast
from contextlib import contextmanager, nullcontext
import random
import re
from transformers import logging
logging.set_verbosity_error()

from models.policy import PolicyNetwork
from models.llm import get_gpt_output
from base_prompt import build_prompt
import utils


def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.benchmark = True

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

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
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


def load_all_models(ckpt, device):
    # - - - - - prepare models - - - - - #
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(ckpt, device)
    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    all_models = (model, autoencoder, text_encoder, diffusion, config)
    return all_models


@torch.no_grad()
def generate_one_image(args, gligen, caption, label, bbox, clip_model=None, clip_processor=None, device=None):
    meta = dict(
        prompt=caption,
        phrases=label,
        locations=bbox,
        alpha_type=[0.3, 0.0, 0.7],
    )
    starting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    model, autoencoder, text_encoder, diffusion, config = gligen

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
    t0 = time.time()
    samples_fake = sampler.sample(S=steps, shape=shape, input=input, uc=uc, guidance_scale=config.guidance_scale,
                                  mask=inpainting_mask, x0=z0)
    samples_fake = autoencoder.decode(samples_fake)
    # print('sampling takes {:.2f}s'.format(time.time() - t0))
    imgs_gen = []
    for sample in samples_fake:
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1, 2, 0) * 255
        sample = Image.fromarray(sample.astype(np.uint8))
        imgs_gen.append(sample)

    return imgs_gen


def convert_xcycwh_to_ltrb(bbox: Union[np.ndarray, FloatTensor]):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def convert_xywh_to_ltrb(bbox: Union[np.ndarray, FloatTensor]):
    # print(bbox)
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


def draw_box(sample, locations, phrases):
    draw = ImageDraw.Draw(sample)
    w, h = sample.size
    for i in range(len(locations)):
        x0, y0, x1, y1 = locations[i]
        x0, y0, x1, y1 = x0 * w, y0 * h, x1 * w, y1 * h 
        draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 255), width=4)
        draw.text((x0, y0-18), phrases[i], (255, 0, 0))

@torch.no_grad()
def run(all_models, meta, config, starting_noise=None):
    model, autoencoder, text_encoder, diffusion, _ = all_models
    # - - - - - prepare batch - - - - - #
    if "keypoint" in meta["ckpt"]:
        batch = prepare_batch_kp(meta, config.batch_size)
    else:
        batch = prepare_batch(meta, config.batch_size)
    
    context = text_encoder.encode(  [meta["prompt"]]*config.batch_size  )
    uc = text_encoder.encode( config.batch_size*[""] )

    # - - - - - sampler - - - - - # 
    alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
    if config.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 250 
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50 

    # - - - - - inpainting related - - - - - #
    inpainting_mask = z0 = None  # used for replacing known region in diffusion process
    inpainting_extra_input = None # used as model input 

    # - - - - - input for gligen - - - - - #
    grounding_input = model.grounding_tokenizer_input.prepare(batch)
    input = dict(
                x = starting_noise, 
                timesteps = None, 
                context = context, 
                grounding_input = grounding_input,
                inpainting_extra_input = inpainting_extra_input
            )


    # - - - - - start sampling - - - - - #
    shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)
    samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=inpainting_mask, x0=z0)
    samples_fake = autoencoder.decode(samples_fake)

    # - - - - - save - - - - - #
    output_folder = os.path.join( args.folder,  meta["save_folder_name"])
    os.makedirs( output_folder, exist_ok=True)

    start = len( os.listdir(output_folder) )
    image_ids = list(range(start,start+config.batch_size))

    imgs_gen = []
    for image_id, sample in zip(image_ids, samples_fake):
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1,2,0) * 255 
        sample = Image.fromarray(sample.astype(np.uint8))
        img_name = meta['save_name']
        sample.save(os.path.join(output_folder, img_name)   )
        imgs_gen.append(sample)

    return imgs_gen



def get_batch_result(scores, cand_examples, all_prompts, args, **kwargs):
    batch_captions = []
    batch_log_prob = []
    batch_bbox_category_pred = []
    batch_bbox_category = []
    batch_imgs_gt = []
    batch_gpt_outputs_raw = []
    ## loop over the examples
    for i in tqdm(range(len(scores)), desc='Calling LLM'):
        # interact with the environment to get rewards, which in our case is to feed the prompt into GPT-3 and evaluate the prediction
        s = scores[i, :].clone().detach()
        s = s.cpu().numpy().tolist()
        # sample shot_pids from the cand_prob distribution
        cids = sorted(range(len(s)), key=lambda i: s[i], reverse=True)[:args.shot_number]
        # reverse shot_pids so more relevant prompt will be put closer to the question
        cids = cids[::-1]
        shot_cand = [cand_examples[cid] for cid in cids]
        # generate the prompt input
        prompt = build_prompt(shot_cand, {'captions': all_prompts[i]}, args)
        # get the output from GPT
        gpt_args = dict(
            engine = args.engine, 
            temperature = args.temperature, 
            max_tokens = args.max_tokens, 
            presence_penalty = args.presence_penalty, 
            frequency_penalty = args.frequency_penalty
        )
        output = get_gpt_output(prompt, **gpt_args)
        # extract the prediction from the output
        prediction = utils.extract_prediction(output)
        categories, bboxes = prediction
        batch_captions.append(all_prompts[i])
        batch_bbox_category_pred.append((bboxes, categories))
        batch_gpt_outputs_raw.append(output)

    return batch_gpt_outputs_raw, batch_bbox_category_pred, cids


def extract_text_feat(prompt_list, tokenizer, model):
    inputs = tokenizer(prompt_list, padding=True, return_tensors="pt").to(model.device)
    feat = model.get_text_features(**inputs)
    return feat


def generate(args, clip, policy_model, gligen, device, is_center=False):
    clip_model, clip_processor, tokenizer = clip
    # extract prompt feature
    all_prompt = [args.prompt]
    emb_prompt = extract_text_feat(all_prompt, tokenizer, clip_model)
    # extract all features from candidate pool
    raw_cand = utils.load_json(args.cand_path)
    ids_cand, examples_cand = raw_cand['id'], raw_cand['data']
    captions_cand = [ex['captions'] for ex in examples_cand]
    emb_cand = extract_text_feat(captions_cand, tokenizer, clip_model)
    # select in-context examples with policy
    emb_cand, emb_prompt = emb_cand.to(device), emb_prompt.to(device)
    embedding_cands = policy_model(emb_cand)  # len(examples_cand) x embedding_size
    embedding_prompt = policy_model(emb_prompt)  # len(prompt_list) x embedding_size
    scores = torch.mm(embedding_prompt, embedding_cands.t())  # len(train_batch) x len(examples_cand)
    batch_gpt_outputs_raw, batch_bbox_category_pred, cids = get_batch_result(scores, examples_cand, all_prompt, args)
    bboxes, categories = [batch_bbox_category_pred[0][0]], [batch_bbox_category_pred[0][1]]

    meta_list = [ 
        dict(
            prompt = all_prompt[i],
            phrases = categories[i], 
            locations = bboxes[i], 
            alpha_type = [0.3, 0.0, 0.7],
        ) for i in range(len(categories))
    ]

    for i, data in enumerate(meta_list):
        if is_center:
            meta_list[i]['locations'] = [convert_xcycwh_to_ltrb(b) for b in data['locations']]
        else:
            meta_list[i]['locations'] = [convert_xywh_to_ltrb(b) for b in data['locations']]
    meta_list = meta_list * args.num_per_prompt
    starting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)

    # generate
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    output_folder = os.path.join(args.folder)
    os.makedirs(output_folder, exist_ok=True)
    with tqdm(total=len(meta_list), desc='Generation') as pbar:
        for i, meta in enumerate(meta_list):
            img_path = os.path.join(output_folder, f"{meta['prompt']}_{i}.jpg")
            imgs = generate_one_image(args, gligen=gligen, caption=meta['prompt'], label=meta['phrases'], bbox=meta['locations'], 
                                clip_model=clip_model, clip_processor=clip_processor, device=device)
            assert len(imgs) == 1
            sample = imgs[0]
            draw_box(sample, meta['locations'], meta['phrases'])
            sample.save(img_path)
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0', help='Used gpt id')
    parser.add_argument("--folder", type=str,  default="generation_samples", help="root folder for output")
    parser.add_argument("--batch_size", type=int, default=1, help="This will overwrite the one in yaml.")
    parser.add_argument("--num_per_prompt", type=int, default=5, help="Number of generated images for per prompt")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="Scale of classifier-free guidance")
    parser.add_argument("--seed", type=int, default=42, help='Random seed')
    parser.add_argument("--in_dim", type=int, default=768, help='Input dimension of Policy network, the same as CLIP dim.')
    parser.add_argument("--prompt", type=str, default='', help='text prompt')
    parser.add_argument('--cand_path', type=str, required=True, help='Path of candidate layout examples')
    parser.add_argument("--policy_ckpt_path", type=str, required=True, help='Path of Policy network weights')
    parser.add_argument("--diff_ckpt_path", type=str, required=True, help='Path of GLIGEN-based relation-aware diffusion model')
    args = parser.parse_args()
    # combine args with args from the training phase
    args_train = utils.load_json(os.path.join(args.policy_ckpt_dir, 'args.txt'))
    args = vars(args)
    args_train.update(args)
    args = argparse.Namespace(**args_train)

    # basic setting
    set_seed(args.seed)
    args.gpu = 0
    device = f'cuda:{args.gpu}'
    is_center = False

    # load clip model
    clip_model = CLIPModel.from_pretrained(args.model_config).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.model_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_config)
    clip = (clip_model, clip_processor, tokenizer)

    # load policy network
    policy_model = PolicyNetwork(model_config=args.model_config, 
                                  in_dim=args.in_dim, 
                                  add_linear=True,
                                  embedding_size=args.embedding_size,
                                  freeze_encoder=True)
    policy_model.linear.load_state_dict(torch.load(args.policy_ckpt_path, map_location='cpu')) 
    policy_model = policy_model.to(device)
    policy_model.eval()

    # load gligen
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(args.diff_ckpt_path, device=device)
    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    config.update(vars(args))
    config = OmegaConf.create(config)
    gligen = [model, autoencoder, text_encoder, diffusion, config]

    # generate
    generate(args, clip, policy_model, gligen, device, is_center=is_center)
    