'''
Get leaderboard predictions for a given CLIP model
'''
import argparse
import numpy as np
import torch
import json
import pprint
import tqdm
import os
import collections
import clip
import accelerate
import random
import subprocess
import pprint
import train_clip as trainlib
from datasets import load_dataset, load_from_disk
from clip_dataset import CLIPDataset, CLIPTEXTAugDataset

def get_args_from_checkpoint_filename(fname):
    kvars = fname.split('/')[-1].split('.pt')[0]
    kvars = kvars.split('~')
    kvars = {x.split('=')[0] : x.split('=')[1] for x in kvars}
    if 'model' in kvars:
        kvars['model'] = kvars['model'].replace('*', '/')

    for k in ['pad', 'split']:
        if k in kvars:
            kvars[k] = int(kvars[k])

    return kvars['task'], kvars['split'], kvars['pad'], kvars['model']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, help='0,1,2,3,4 are cross-val splits 5 is leaderboard', default=5)
    parser.add_argument('--clip_model_path',
                        default='zero_shot',
                        help='either a path to model, or "zero_shot" for zero shot evaluation')

    parser.add_argument('--batch_size',
                        default=100,
                        type=int)

    parser.add_argument('--prefix',
                        default=None,
                        type=str,
                        help='if this prefix is set, it will be appended to the input.')

    parser.add_argument('--dataset_path',
                        default="/data/mjjung/The_new_yorker_caption_contest",
                        type=str,
                        help='if this prefix is set, it will be appended to the input.')

    ### these arguments will be autospecified if clip_model_path is specified, else, for zero shot, you need to specify these.
    parser.add_argument('--task',
                        default='matching',
                        choices=['matching', 'ranking'],
                        type=str,
                        help='what task are we looking at?')

    parser.add_argument('--pad',
                        default=1,
                        type=int,
                        help='if 0 we will do standard center crop, if 1 we will do pad.')

    parser.add_argument('--clip_model',
                        default='ViT-L/14@336px',
                        type=str,
                        choices=['ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'ViT-B/16', 'RN50x16', 'RN50x64', 'ViT-L/14@336px', 'ViT-L/14'])

    args = parser.parse_args()

    if args.clip_model_path != 'zero_shot':
        args.task, args.split, args.pad, args.clip_model = get_args_from_checkpoint_filename(args.clip_model_path)

    if args.clip_model_path != 'zero_shot':
        args.zero_shot_mode = False
        args.prefix = '' # assume fine-tuned models are trained with no prefix.
        args.output = args.clip_model_path.replace('.pt', '~results.json')
    else:
        print('zero-shot mode')
        args.zero_shot_mode = True
        if not args.prefix:
            args.prefix = ''
        else:
            args.prefix = args.prefix.strip()
        args.output = 'task={}~split={}~prefix={}~pad={}~model={}~results.json'.format(
            args.task,
            5,
            '+'.join(args.prefix.strip().split()),
            args.pad,
            args.clip_model.replace('/', '*'))
        args.prefix = args.prefix + ' '

    print('padding={}, backbone={}'.format(args.pad, args.clip_model))
    args.use_accelerate = False

    print('writing all results to {}'.format(args.output))

    return args


def main():
    args = parse_args()
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(args.clip_model, jit=False)
    model.float()
    model.eval()

    zero_shot_clip, _ = clip.load(args.clip_model, device=0, jit=False)
    zero_shot_clip.eval()

    # load model #
    if 'zero_shot' not in args.clip_model_path:
        print('Getting model weights from {}'.format(args.clip_model_path))
        state = torch.load(args.clip_model_path)
        state['model_state_dict'] = {k.replace('module.', '') : v for k, v in state['model_state_dict'].items()}
        model.load_state_dict(state['model_state_dict'])
    else:
        print('doing zero shot with {}!'.format(args.clip_model))

    model.eval()
    try:
        args.input_resolution = model.visual.input_resolution
    except:
        args.input_resolution = model.input_resolution


    try:
        data = load_from_disk(args.dataset_path)
    except:
        if args.task == 'matching':
            split_name = 'matching_from_pixels' if args.split in [0, 5] else 'matching_from_pixels_{}'.format(
                args.split)
        elif args.task == 'ranking':
            split_name = 'ranking_from_pixels' if args.split in [0, 5] else 'ranking_from_pixels_{}'.format(args.split)

        data = load_dataset("jmhessel/newyorker_caption_contest", split_name)
        data.save_to_disk(args.dataset_path)

    train, val, test = data['train'], data['validation'], data['test']
    if args.split == 5:
        train = list(train) + list(val)
        val = test
        test = []

    print('train/val/test datapoints: {}/{}/{}'.format(*map(len, [train, val, test])))

    try:
        logit_scale = model.module.logit_scale
    except:
        logit_scale = model.logit_scale


    if args.task == 'matching':
        train = [trainlib.convert_matching(t, args) for t in train]
        val = [trainlib.convert_matching(t, args) for t in val]
    elif args.task == 'ranking':
        train = [trainlib.convert_quality(t, args) for t in train]
        val = [trainlib.convert_quality(t, args) for t in val]
    else:
        raise NotImplementedError

    trainlib.add_prefix(train, args)
    trainlib.add_prefix(val, args)

    # train_loader = CLIPDataset(train, args, training=True)
    train_loader = CLIPTEXTAugDataset(train, args, training=True)
    # val_loader = CLIPDataset(val, args, training=False)
    val_loader = CLIPTEXTAugDataset(val, args, training=False)

    train_loader = torch.utils.data.DataLoader(
        train_loader, shuffle=True, batch_size=args.batch_size, num_workers=4,
    )

    val_loader = torch.utils.data.DataLoader(
        val_loader, shuffle=False, batch_size=args.batch_size, num_workers=4
    )

    # bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    bar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))

    logits_gt, logits_non_gt = [], []
    results = {}
    running_sum_accs = 0

    for i, batch in bar:
        with torch.no_grad():
            meta, batch = trainlib.batch_to_device(batch, 'train', args)
            n_choice = batch['choices'].shape[1]
            batch['choices'] = batch['choices'].reshape((-1, 77))
            image_features, text_features = trainlib.clip_forward(model, batch['image'], batch['choices'])
            text_features = text_features.reshape((image_features.shape[0], n_choice, -1))
            image_features = torch.unsqueeze(image_features, 1)
            logits = logit_scale.exp() * (image_features * text_features).sum(2)
            preds = logits.argmax(1).cpu().numpy()

            running_sum_accs += np.sum(preds == batch['labels'].detach().cpu().numpy())
            gt_labels = batch['labels'].tolist()
            non_gt_labels = []
            for label in gt_labels:
                non_gt_labels.extend([i for i in range(5) if i != label])

            indices = torch.arange(len(logits))
            neg_indices = [[i]*4 for i in range(len(logits))] # the total number of candidate answer is 5.
            neg_indices = torch.flatten(torch.tensor(neg_indices))
            gt_logits = logits[indices, gt_labels]
            non_gt_logits = logits[neg_indices, non_gt_labels]

            logits_gt.extend(gt_logits)
            logits_non_gt.extend(non_gt_logits)
            # break

    print('acc = {:.3f}'.format(running_sum_accs / len(val_loader)))
    print('gt logit = {:.3f}'.format(sum(logits_gt) / len(logits_gt)))
    print('non gt logit = {:.3f}'.format(sum(logits_non_gt) / len(logits_non_gt)))


if __name__ == '__main__':
    main()