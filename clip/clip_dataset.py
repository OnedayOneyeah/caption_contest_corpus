import argparse
import numpy as np
import time
import json
import pprint
import PIL
from PIL import Image, ImageDraw
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomGrayscale, ColorJitter
import tempfile
from tqdm import tqdm
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
# from utils import dict_to_markdown, mkdirp
from utils import save_json, load_json

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        self.mode = args.mode

        self.training = training
        if self.args.pad:
            self.preprocess = self._transform_train(args.input_resolution) if self.training else self._transform_test(args.input_resolution)
        else:
            self.preprocess = self._transform_train_pad(args.input_resolution) if self.training else self._transform_test_pad(args.input_resolution)

        if self.mode in ['keywords', 'antonym']:
            # keyword extraction
            import nltk
            from rake_nltk import Rake
            nltk.download('punkt')
            nltk.download('stopwords')

            def extract_keyword(sent):
                
                rake_nltk_var = Rake()
                rake_nltk_var.extract_keywords_from_text(sent)
                extracted_keyword = rake_nltk_var.get_ranked_phrases()[0] # keyword extraction, phrases format.

                return extracted_keyword
              
        if self.mode == "rephrase":
            rephrase_json_path = 'rephrase_train.json' if training else 'rephrase_val.json'
            if os.path.exists(rephrase_json_path):
                self.rephrase = load_json(rephrase_json_path)
            else:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
                prompt = 'Re-write the sentence while maintaining its semantic meanings: '

                try:
                    path = '/data/mjjung/storage/falcon-7b'
                    rephraser = AutoModelForCausalLM.from_pretrained(path)
                except:
                    rephraser = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct")

                rephraser.to(device)
                rephraser.eval()

                self.rephrase = dict()
                prompt = 'Re-write the sentence while maintaining its semantic meanings: '
                for c_data in tqdm(self.data, total=len(self.data), desc='rephrase the sentences..'):
                    gt_sent = c_data['choices'][c_data['label']]
                    input_text = prompt + gt_sent
                    token = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

                    with torch.no_grad():
                        output = rephraser.generate(token, max_new_tokens=100)
                    sent = tokenizer.decode(output[0])
                    new_gt_sent = sent.split('\n')[1].split('<|endoftext|>')[0]
                    self.rephrase[c_data['contest_number']] = new_gt_sent
                save_json(self.rephrase, rephrase_json_path)

        elif self.mode == 'keywords':
            keywords_json_path = 'keywords.json'
            if os.path.exists(keywords_json_path):
                self.keywords = load_json(keywords_json_path)
            else:
                self.keywords = dict()
                for c_data in tqdm(self.data, total = len(self.data), desc = 'extract the keywords from sentences..'):
                    gt_sent = c_data['choices'][c_data['label']]
                    keywords = extract_keyword(gt_sent)
                    self.keywords[c_data['contest_number']] = keywords
                save_json(self.keywords, keywords_json_path)

        elif self.mode == 'antonym':
            antonym_json_path = 'antonym.json'
            if os.path.exists(antonym_json_path):
                self.antonym = load_json(antonym_json_path)
            else:
                self.antonym = dict()
                prompt = 'Re-write the sentence while replacing the keyword with its antonym.'

                for c_data in tqdm(self.data, total = len(self.data), desc = 'change the keywords to their antonyms..'):
                    gt_sent = c_data['choices'][c_data['label']]
                    keyword = extract_keyword(gt_sent)
                    input_text = f'sentence: {gt_sent}, keyword: {keyword} ' + prompt
                    token = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
                    with torch.no_grad():
                        output = self.rephraser.generate(token, max_new_tokens = 100)
                    sent = self.tokenizer.decode(output[0])
                    ant_sent = sent.split('\n')[1].split('<|endoftext|>')[0]
                    self.antonym[c_data['contest_number']] = ant_sent
                save_json(self.antonym, antonym_json_path)

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
        if self.mode == 'rephrase':
            new_gt_sent = self.rephrase[c_data['contest_number']]
            c_data['choices'][c_data['label']] = new_gt_sent
        elif self.mode == 'keywords':
            keywords = self.keywords[c_data['contest_number']]
            c_data['choices'][c_data['label']] = keywords
        elif self.mode =='antonym':
            antonym = self.antonym[c_data['contest_number']]
            c_data['choices'][c_data['label']] = antonym

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