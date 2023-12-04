import json
import os
from tqdm import tqdm
from datetime import datetime
import argparse
import random
import numpy as np

import torch
import torch.nn.functional as F
from transformers import logging
logging.set_verbosity_error()
import tensorboard_logger as tb_logger

from models.policy import PolicyNetwork, Reward
import utils
from data import COCO2014, collate_fn, load_data
from base_prompt import build_prompt
from GLIGEN.interface import generate_batch_images, load_all_models
from models.llm import get_gpt_output


def get_batch_reward_loss(reward_model, diff_model_lst, scores, cand_examples, train_batch, batch_imgs_gt_old, 
                            args, id_map=None, **kwargs):
    batch_loss = 0
    batch_reward = 0
    batch_imgs_pred = []
    batch_captions = []
    batch_log_prob = []
    batch_bbox_pred, batch_category_pred = [], []
    batch_bbox_category_pred = []
    batch_bbox_category = []
    
    batch_imgs_gt = []
    ## loop over the training examples
    for i in tqdm(range(len(scores))):
        # interact with the environment to get rewards, which in our case is to feed the prompt into GPT-3 and evaluate the prediction
        cand_prob = scores[i, :].clone().detach()
        cand_prob = cand_prob.cpu().numpy()
        cand_prob = np.nan_to_num(cand_prob, nan=0.000001)  # replace np.nan with 0
        cand_prob /= cand_prob.sum()  # make probabilities sum to 1

        # sample shot_pids from the cand_prob distribution
        # cand_prob = None # ablation: random selection
        cids = np.random.choice(range(len(cand_prob)), args.shot_number, p=cand_prob, replace=False)

        # reverse shot_pids so more relevant prompt will be put closer to the question
        cids = cids[::-1]

        if id_map is not None:
            pool_ids = cids.copy()
            cids = id_map[i][pool_ids]
        else:
            pool_ids = cids

        shot_cand = [cand_examples[cid] for cid in cids]

        # generate the prompt input
        prompt = build_prompt(shot_cand, train_batch[i], args)

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

        if len(categories) == 0:
            continue

        batch_captions.append(train_batch[i]['captions'])
        batch_imgs_gt.append(batch_imgs_gt_old[i])
        batch_bbox_category_pred.append((bboxes, categories))
        batch_bbox_pred.append(bboxes)
        batch_category_pred.append(categories)
        batch_bbox_category.append((utils.center2lefttop(train_batch[i]['bbox']), train_batch[i]['label']))

        log_prob = 0
        for pid in pool_ids:
            log_prob += torch.log(scores[i, pid])
        batch_log_prob.append(log_prob)

    batch_imgs_pred = generate_batch_images(diff_model_lst, batch_captions, batch_category_pred, batch_bbox_pred, 
                            clip_model=reward_model.model, clip_processor=reward_model.processor, device=args.device)

    batch_reward = reward_model(batch_captions, batch_imgs_pred, batch_imgs_gt, batch_bbox_category_pred, batch_bbox_category)
    batch_log_prob = torch.stack(batch_log_prob)
    batch_loss = (-batch_log_prob * batch_reward).sum()
    batch_reward = batch_reward.sum().item()

    return cids, batch_reward, batch_loss

def resume(ckpt_dir, model, optimizer, lr_scheduler):
    f_names = os.listdir(ckpt_dir)
    max_old_epo= 0
    for fn in f_names:
        if fn[:5] == 'state':
            s = int(fn.split('.')[0].split('_')[-1])
            if s > max_old_epo:
                max_old_epo = s
    
    weight_ckpt = torch.load(os.path.join(ckpt_dir, f'ckpt_{max_old_epo}.pt'))
    state_ckpt = torch.load(os.path.join(ckpt_dir, f'state_{max_old_epo}.pt'))
    model.linear.load_state_dict(weight_ckpt)
    optimizer.load_state_dict(state_ckpt['optimizer'])  
    return max_old_epo


def policy_gradient_train(policy_model, reward_model, diff_model_lst, train_examples, cand_examples, 
                            train_ids, cand_ids, feats_caps_train, feats_caps_cand, args, **kwargs):
    # REINFORCE
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma) 

    # resume
    start_epoch = 0
    if len(args.resume) > 0:
        max_old_epo = resume(args.resume, policy_model, optimizer, lr_scheduler)
        start_epoch = max_old_epo + 1

    # use new learning rate
    for params in optimizer.param_groups: 
        params['lr'] = args.lr


    train_dataset = COCO2014(train_examples, feats_caps_train, args.img_dir)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn, 
            pin_memory=True)


    reward_history = []
    loss_history = []

    total_reward_history = []  # epoch based
    total_loss_history = []  # epoch based

    STOP_FLAG = False
    Eiters = 0

    for epoch in range(start_epoch, start_epoch + args.epochs):
        logger.write(f"Epoch: {epoch}")

        total_train_reward = 0
        total_train_loss = 0

        # We can simply set the batch_size to len(train_data) in few-shot setting.
        for batch_i, data_batch in enumerate(train_loader):
            logger.write(f"Batch: {batch_i}")

            train_batch, train_batch_emb_inp, train_batch_ids, train_batch_imgs = data_batch
            train_batch_emb_inp = train_batch_emb_inp.to(args.device)
            feats_caps_cand = feats_caps_cand.to(args.device)

            # We need to encode cands again every time we update the network
            embedding_cands = policy_model(feats_caps_cand)  # len(cand_examples) x embedding_size
            embedding_ctxt = policy_model(train_batch_emb_inp)  # len(train_batch) x embedding_size

            scores = torch.mm(embedding_ctxt, embedding_cands.t())  # len(train_batch) x len(cand_examples)
            sort_idx = None
            scores = F.softmax(scores/args.policy_temperature, dim=1)  # len(train_batch) x len(cand_examples)

            cids, reward, loss = get_batch_reward_loss(reward_model, diff_model_lst, scores, cand_examples, train_batch, 
                                                        train_batch_imgs, args, id_map=sort_idx)

            logger.write(f"cids for sample[-1] in batch: {cids}")
            logger.write(f"Cand prob for sample[-1] in batch: {[round(x,5) for x in scores[-1, :].tolist()]}")
            logger.write(f"### reward for the batch: {reward}")
            logger.write(f"### loss for the batch: {loss}\n")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for each iteration/batch
            total_train_reward += reward
            total_train_loss += loss.item()

            reward_history.append(reward)
            loss_history.append(loss.item())
            tb_logger.log_value('reward', reward, step=Eiters)
            tb_logger.log_value('loss', loss.item(), step=Eiters)
            tb_logger.log_value('LR', optimizer.param_groups[0]['lr'], step=Eiters)
            Eiters += 1

            if np.isnan(loss.item()):
                STOP_FLAG = True
                break

        # for each epoch
        total_reward_history.append(total_train_reward)
        total_loss_history.append(total_train_loss)

        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)

        best_reward_epoch = total_reward_history.index(best_reward) + start_epoch
        best_loss_epoch = total_loss_history.index(best_loss) + start_epoch

        logger.write("============================================")
        logger.write(f"### Epoch: {epoch} / {args.epochs}")
        logger.write(f"### Total reward: {total_train_reward}, " + f"Total loss: {round(total_train_loss,5)}, " +
                     f"Best reward: {best_reward} at epoch {best_reward_epoch}, " +
                     f"Best loss: {round(best_loss, 5)} at epoch {best_loss_epoch}\n")

        # save every epoch
        ckpt_file = os.path.join(args.ckpt_path, f"ckpt_{epoch}.pt")
        torch.save(policy_model.linear.state_dict(), ckpt_file)
        state_file = os.path.join(args.ckpt_path, f"state_{epoch}.pt")
        state = {'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict()}
        torch.save(state, state_file)
        logger.write(f"saved the ckpt to {ckpt_file} and {state_file}")

        # save best epoch
        if epoch == best_reward_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_reward.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best reward ckpt to {ckpt_file}")

        if epoch == best_loss_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_loss.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best loss ckpt to {ckpt_file}")

        # save reward and loss history
        history = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "total_reward_history": total_reward_history,
            "total_loss_history": total_loss_history,
        }
        history_file = os.path.join(args.ckpt_path, "history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, separators=(',', ': '))

        # print cache info
        logger.write(get_gpt_output.cache_info())
        logger.write("============================================\n")
        lr_scheduler.step()

        if STOP_FLAG:
            break

    # save in the end
    ckpt_file = os.path.join(args.ckpt_path, "ckpt_final.pt")
    torch.save(policy_model.linear.state_dict(), ckpt_file)


def parse_args():
    parser = argparse.ArgumentParser()

    # User options
    parser.add_argument('--exp', type=str, default='exp0', help='Expeirment name')
    parser.add_argument('--shot_number', type=int, default=2, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=53, help='random seed')
    parser.add_argument('--resume', type=str, default='', help='Path of resume ckpt, empty means no resuming')
    # Data
    parser.add_argument('--img_dir', type=str, required=True, help='Path of images')
    parser.add_argument('--sampled_data_dir', type=str, default='./data', help='Path of DATA directory')
    parser.add_argument('--train_number', type=int, default=64, help='Number of training samples.')
    parser.add_argument('--cand_number', type=int, default=32, help='Number of candidate prompts.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers to load data.')
    # GPT settings
    parser.add_argument('--engine', type=str, default='gpt-3.5-turbo', choices=['text-davinci-002', 'gpt-3.5-turbo'], help='GPT version')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature factor for GPT')
    parser.add_argument('--max_tokens', type=int, default=512, help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty factor for GPT')
    parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty factor for GPT')
    # Policy gradient settings
    parser.add_argument('--gpu', type=str, default='0', help='Gpu id')
    parser.add_argument('--model_config', type=str, default='openai/clip-vit-large-patch14', help='Version of CLIP')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of policy network.')
    parser.add_argument('--lr_step_size', type=int, default=20, help='Step size for lr scheduler.')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Decay rate for lr scheduler.')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs.')
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--batch_size', type=int, default=8, help='Policy network training batch size. Set to train_number by default.')
    parser.add_argument('--policy_temperature', type=float, default=1.0, help='Temperature factor for policy network') 
    parser.add_argument('--diff_ckpt', type=str, required=True, help='Path of diffusion model ckpt')
    parser.add_argument('--ckpt_root', type=str, default='./checkpoints', help='Root path of saved ckpt')
    parser.add_argument('--aesthetic_ckpt', type=str, required=True, help='Path of aesthetic predictor ckpt')
    
    args = parser.parse_args()

    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("_%Y_%m_%d_%H_%M_%S")
    args.exp = args.exp + currentTime

    # print and save the args
    args.ckpt_path = os.path.join(args.ckpt_root, args.exp)
    utils.create_dir(args.ckpt_path)
    _logger = utils.Logger(args.ckpt_path + '/args.txt')

    print('====Input Arguments====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))
    return args


if __name__ == '__main__':
    args = parse_args()
    utils.set_seed(args.seed)

    ## Model
    policy_model = PolicyNetwork(model_config=args.model_config, 
                                  in_dim=768, 
                                  add_linear=True,
                                  embedding_size=args.embedding_size,
                                  freeze_encoder=True)

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
    args.device = device
    policy_model = policy_model.to(device)

    reward_model = Reward(args.model_config, args.aesthetic_ckpt, args, device=device)
    reward_model = reward_model.to(device)

    diff_model_lst = load_all_models(args.diff_ckpt, device)

    ## Data
    train_examples, cand_examples, train_ids, cand_ids, feats_caps_train, feats_caps_cand = load_data(args, reward_model)

    ## TRAINING
    logger = utils.Logger(os.path.join(args.ckpt_path, 'log.txt'))
    gpt_logger = utils.Logger(os.path.join(args.ckpt_path, 'gpt_log.txt'))
    tb_logger.configure(args.ckpt_path, flush_secs=5)
    policy_gradient_train(policy_model, reward_model, diff_model_lst, train_examples, cand_examples, 
                            train_ids, cand_ids, feats_caps_train, feats_caps_cand, args)



