import argparse
import numpy as np
import time
import json
import pprint
import PIL
from PIL import Image, ImageDraw
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomGrayscale, ColorJitter
import tempfile
import tqdm
import os
import collections
import clip
import torch
import torchvision.transforms.functional as F
import accelerate
import random
import subprocess
import pprint
from datasets import load_dataset, load_from_disk
from utils import dict_to_markdown, mkdirp

import requests

API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
API_TOKEN= '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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}
prompt = 'Please rewrite the following sentence while maintaining its semantic meanings: '

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def rewrite(sentence):
    output = query({
        "inputs": f"{prompt} {sentence}",
    })
    print(output)
    return output[0]['generated_text'].split('\n')[1]


class SquarePad:
    # https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/9
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, data, args, training=False):
        self.args = args
        self.data = data

        self.training = training
        if self.args.pad:
            self.preprocess = self._transform_train(args.input_resolution) if self.training else self._transform_test(args.input_resolution)
        else:
            self.preprocess = self._transform_train_pad(args.input_resolution) if self.training else self._transform_test_pad(args.input_resolution)

    def _transform_train_pad(self, n_px):
        return Compose([
            SquarePad(),
            Resize(n_px, interpolation=Image.BICUBIC),
            RandomHorizontalFlip(),
            RandomGrayscale(),
            ColorJitter(brightness=.5, hue=.3),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_test_pad(self, n_px):
        return Compose([
            SquarePad(),
            Resize(n_px, interpolation=Image.BICUBIC),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_train(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            RandomCrop(n_px),
            RandomHorizontalFlip(),
            RandomGrayscale(),
            ColorJitter(brightness=.5, hue=.3),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def image_to_torch_tensor(self, image):
        image = self.preprocess(image)
        return image

    def __getitem__(self, idx):
        c_data = self.data[idx]
        if 'filepath' in c_data:
            image = Image.open(c_data['filepath'])
        else:
            image = c_data['image']
        choices = clip.tokenize(c_data['choices'], truncate=True)
        image = self.image_to_torch_tensor(image)
        to_ret = {'image': image, 'choices': choices, 'label': c_data['label'], 'meta': dict(raw_choices=c_data['choices'],
                                                                                             contest_number=c_data['contest_number'])}
        if 'instance_id' in c_data:
            to_ret['instance_id'] = c_data['instance_id']
        return to_ret

    def __len__(self):
        return len(self.data)


class CLIPTEXTAugDataset(torch.utils.data.Dataset):
    def __init__(self, data, args, training=False):
        self.args = args
        self.data = data

        self.training = training
        if self.args.pad:
            self.preprocess = self._transform_train(args.input_resolution) if self.training else self._transform_test(args.input_resolution)
        else:
            self.preprocess = self._transform_train_pad(args.input_resolution) if self.training else self._transform_test_pad(args.input_resolution)

    def _transform_train_pad(self, n_px):
        return Compose([
            SquarePad(),
            Resize(n_px, interpolation=Image.BICUBIC),
            RandomHorizontalFlip(),
            RandomGrayscale(),
            ColorJitter(brightness=.5, hue=.3),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_test_pad(self, n_px):
        return Compose([
            SquarePad(),
            Resize(n_px, interpolation=Image.BICUBIC),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_train(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            RandomCrop(n_px),
            RandomHorizontalFlip(),
            RandomGrayscale(),
            ColorJitter(brightness=.5, hue=.3),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def image_to_torch_tensor(self, image):
        image = self.preprocess(image)
        return image

    def __getitem__(self, idx):
        c_data = self.data[idx]
        try:
            new_gt_sent = rewrite(c_data['choices'][c_data['label']])
            c_data['choices'][c_data['label']] = rewrite(c_data['choices'][c_data['label']])
        except:
            print('error with the sentence:', c_data['choices'][c_data['label']])
        if 'filepath' in c_data:
            image = Image.open(c_data['filepath'])
        else:
            image = c_data['image']
        choices = clip.tokenize(c_data['choices'], truncate=True)
        image = self.image_to_torch_tensor(image)
        to_ret = {'image': image, 'choices': choices, 'label': c_data['label'], 'meta': dict(raw_choices=c_data['choices'],
                                                                                             contest_number=c_data['contest_number'])}
        if 'instance_id' in c_data:
            to_ret['instance_id'] = c_data['instance_id']
        return to_ret

    def __len__(self):
        return len(self.data)