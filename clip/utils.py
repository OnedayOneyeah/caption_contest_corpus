import os
import json
import zipfile
import numpy as np
import pickle
from collections import OrderedDict, Counter
import pandas as pd
import torch
import PIL

def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def dict_to_markdown(d, max_str_len=120):
    # convert list into its str representation
    d = {k: v.__repr__() if isinstance(v, list) else v for k, v in d.items()}
    # truncate string that is longer than max_str_len
    if max_str_len is not None:
        d = {k: v[-max_str_len:] if isinstance(v, str) else v for k, v in d.items()}
    return pd.DataFrame(d, index=[0]).transpose().to_markdown()

def mkdirp(p):
    # if not os.path.exists(p):
    os.makedirs(p, exist_ok=True)

def clip_forward(model, image, text):

    if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.parallel.DataParallel):
        image_features = model.module.encode_image(image)
        text_features = model.module.encode_text(text)
    else:
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return image_features, text_features


def add_prefix(instances, args):
    for inst in instances:
        inst['choices'] = [(args.prefix + ch).strip() for ch in inst['choices']]


def batch_to_device(batch, mode, args):
    image, choices, labels, meta = batch['image'], batch['choices'], batch['label'], batch['meta']
    if not args.use_accelerate or mode == 'val':
        image, choices, labels = map(
            lambda x: x.to(args.device),
            [image, choices, labels])

    return meta, dict(zip(['image', 'choices', 'labels'], [image, choices, labels]))


def convert_matching(inst, args, leaderboard_mode=False):
    '''
    standardizes into matching format
    '''
    new_inst = {}
    if leaderboard_mode:
        new_inst['choices'] = [inst['choices'][l] for l in 'ABCDE']
        new_inst['label'] = 0 # dummy
        new_inst['contest_number'] = 0 # dummy
        new_inst['instance_id'] = inst['instance_id']
    else:
        new_inst['choices'] = inst['caption_choices']
        new_inst['label'] = 'ABCDE'.index(inst['label'])
        new_inst['contest_number'] = inst['contest_number']

    if isinstance(inst['image'], str):
        new_inst['filepath'] = inst['image']
    elif isinstance(inst['image'], PIL.JpegImagePlugin.JpegImageFile):
        new_inst['image'] = inst['image']
    else:
        new_inst['filepath'] = inst['image']['path']

    return new_inst


def convert_quality(inst, args, leaderboard_mode=False):
    '''
    standardizes into ranking format

    if leaderboard mode, assume the input inst is from the leaderboard format.
    '''
    new_inst = {}
    if leaderboard_mode:
        new_inst['choices'] = [inst['choices'][l] for l in 'AB']
        new_inst['label'] = 0 # dummy
        new_inst['contest_number'] = 0 # dummy
        new_inst['instance_id'] = inst['instance_id']
    else:
        new_inst['choices'] = inst['caption_choices']
        new_inst['label'] = 'AB'.index(inst['label'])
        new_inst['contest_number'] = inst['contest_number']


    if isinstance(inst['image'], str):
        new_inst['filepath'] = inst['image']
    elif isinstance(inst['image'], PIL.JpegImagePlugin.JpegImageFile):
        new_inst['image'] = inst['image']
    else:
        new_inst['filepath'] = inst['image']['path']

    return new_inst