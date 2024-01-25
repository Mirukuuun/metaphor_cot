import os
import math
import torch
import numpy as np
import pickle as pkl

from PIL import Image
import torchvision.transforms as T
from src.utils import prompt_for_entity, prompt_for_metaphor_t5base_llava
from transformers import AutoTokenizer, AutoImageProcessor
from torch.utils.data import Dataset, DataLoader
import random
import nltk


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data_length = 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MetaphorDataLoader:
    def __init__(self, config):
        self.config = config
        config.preprocessor = Preprocessor(config)
        proxies = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, proxies=proxies)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            # T.RandomCrop((384,384)),
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.special_token_mapping = {
            '<mask>': self.tokenizer.mask_token_id,
            '[MASK]': self.tokenizer.mask_token_id,
            '[mask]': 0
        }
        self.mask_token = {
            'roberta-base': '<mask>',
            'roberta-large': '<mask>',
            'google/flan-t5-base': '[mask]',
            'google/flan-t5-large': '[mask]',
            'bert-base-uncased': '[MASK]',
        }
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def worker_init(self, worked_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_data(self):
        cfg = self.config
        path = os.path.join(self.config.preprocessed_dir,
                            '{}_{}_{}.pkl'.format(cfg.data_name, cfg.model_size, cfg.model_path).replace('/', '-'))
        if os.path.exists(path):
            self.data = pkl.load(open(path, 'rb'))
        else:
            self.data = self.config.preprocessor.forward()
            pkl.dump(self.data, open(path, 'wb'))

        train_data, valid_data, test_data = self.data[:3]
        # train_data, valid_data, test_data = train_data.cuda(device=0), valid_data.cuda(device=0), test_data.cuda(device=0)
        self.config.word_dict = self.data[-1]

        load_data = lambda dataset: DataLoader(MyDataset(dataset), num_workers=0, worker_init_fn=self.worker_init,
                                               shuffle=self.config.shuffle, batch_size=self.config.batch_size,
                                               collate_fn=self.collate_fn)
        train_loader, valid_loader, test_loader = map(load_data, [train_data, valid_data, test_data])
        train_loader.data_length, valid_loader.data_length, test_loader.data_length = math.ceil(
            len(train_data) / self.config.batch_size), \
                                                                                      math.ceil(
                                                                                          len(valid_data) / self.config.batch_size), \
                                                                                      math.ceil(
                                                                                          len(test_data) / self.config.batch_size)
        # print('train len:', train_loader.data_length)
        res = [train_loader, valid_loader, test_loader]

        return res, self.config

    def collate_fn(self, data):
        ids, input_texts, input_labels, input_captions, output_template, text_attrs, image_attrs, target, source, image_prefix = zip(
            *data)
        # ids, input_texts, input_labels, input_captions, output_template, text_attrs, image_attrs, target, source = zip(*data)
        step_one_output = None
        if self.config.reasoning == 'prompt' or self.config.reasoning == 'mprompt' or self.config.reasoning == 'extract':
            new_tokens = []
            mask_pos = []
            context_ids = []
            # print('input_texts:', input_texts)
            for i, line in enumerate(input_texts):
                # print('i:{}, text:{}, caption:{}'.format(i, line, input_captions[i]))
                context, prompt = prompt_for_entity(line, input_captions[i], image_attrs[i])
                context_ids.append(context)
                new_tokens.append(prompt)
            truncation = not self.config.cot
            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           truncation=truncation,
                                                           max_length=self.config.max_length)
            context_ids = self.tokenizer.batch_encode_plus(context_ids, return_tensors='pt',
                                                               padding=True,
                                                               max_length=self.config.max_length)
            context_ids = context_ids.data
            batch_input = batch_input.data

            mask_pos = [0]
            mask_labels = [0]
            template = output_template
            output_template = [self.config.label_list[int(w)] for w in input_labels]

            batch_output = self.tokenizer.batch_encode_plus(output_template, max_length=3, padding=True,
                                                            truncation=truncation,
                                                            return_tensors="pt").data
            # print(batch_output)
            if self.config.reasoning == 'mprompt' or self.config.reasoning == 'extract':
                # image_processor = AutoImageProcessor.from_pretrained(self.config.vision_model_path)
                image_path = self.config.image_path
                images = []
                for i, id in enumerate(ids):
                    path = os.path.join(image_path, id + '.jpg')
                    # print("path:",path)
                    try:
                        image = Image.open(path).convert('RGB')
                        image = self.transform(image)
                        images.append(image)
                    except Exception:
                        print('no image:', path)
                        image = torch.zeros([3, 224, 224])
                        images.append(image)

            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'mask_labels': torch.tensor(mask_labels),
                'mask_pos': torch.tensor(mask_pos),
                'context_ids': context_ids['input_ids'],
                'target': target,
                'source': source,
                'ids': ids
            }
            if self.config.reasoning == 'mprompt':
                # print(image_prefix)
                res['image_prefix'] = torch.stack(image_prefix)
            if self.config.reasoning == 'mprompt' or self.config.reasoning == 'extract':
                res['image_ids'] = torch.tensor([item.cpu().detach().numpy() for item in images]).to(self.config.device)
                # res['image_ids'] = images
            # print(res['image_prefix'][0][0])
            for k, v in res.items():
                if k != 'target' and k != 'source' and k != 'ids' and k != 'image_ids':
                    res[k] = v.cuda() if self.config.multi_gpu else v.to(self.config.device)
                else:
                    res[k] = v
            return res
        else:
            raise 'choose correct reasoning mode'


class MetaphorLLaVADataLoader:
    def __init__(self, config):
        self.config = config
        config.preprocessor = Preprocessor(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            # T.RandomCrop((384,384)),
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.special_token_mapping = {
            '<mask>': self.tokenizer.mask_token_id,
            '[MASK]': self.tokenizer.mask_token_id,
            '[mask]': 0
        }
        self.mask_token = {
            'roberta-base': '<mask>',
            'roberta-large': '<mask>',
            'google/flan-t5-base': '[mask]',
            'google/flan-t5-large': '[mask]',
            'bert-base-uncased': '[MASK]',
        }
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def worker_init(self, worked_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_data(self):
        cfg = self.config
        path = os.path.join(self.config.preprocessed_dir,
                            '{}_{}_{}.pkl'.format(cfg.data_name, cfg.model_size, cfg.model_path).replace('/', '-'))
        if os.path.exists(path):
            self.data = pkl.load(open(path, 'rb'))
        else:
            self.data = self.config.preprocessor.forward()
            pkl.dump(self.data, open(path, 'wb'))

        train_data, valid_data, test_data = self.data[:3]
        self.config.word_dict = self.data[-1]

        load_data = lambda dataset: DataLoader(MyDataset(dataset), num_workers=0, worker_init_fn=self.worker_init,
                                               shuffle=self.config.shuffle, batch_size=self.config.batch_size,
                                               collate_fn=self.collate_fn)
        train_loader, valid_loader, test_loader = map(load_data, [train_data, valid_data, test_data])
        train_loader.data_length, valid_loader.data_length, test_loader.data_length = math.ceil(
            len(train_data) / self.config.batch_size), \
                                                                                      math.ceil(
                                                                                          len(valid_data) / self.config.batch_size), \
                                                                                      math.ceil(
                                                                                          len(test_data) / self.config.batch_size)
        # print('train len:', train_loader.data_length)
        res = [train_loader, valid_loader, test_loader]

        return res, self.config

    def collate_fn(self, data):
        ids, input_texts, input_labels, input_captions, output_template, text_attrs, image_attrs, target, source, image_prefix, cot_result = zip(
            *data)
        # ids, input_texts, input_labels, input_captions, output_template, text_attrs, image_attrs, target, source = zip(*data)
        new_tokens = []
        mask_pos = []
        # print('input_texts:', input_texts)
        for i, line in enumerate(input_texts):
            # print('i:{}, text:{}, caption:{}'.format(i, line, input_captions[i]))
            if self.config.model_path.startswith('google'):
                prompt = prompt_for_metaphor_t5base_llava(line, input_captions[i],
                                                    self.mask_token[self.config.model_path],
                                                    cot_result[i], image_attrs[i])
            else:
                if self.config.special:
                    prompt = prompt_for_metaphor_special(line, input_captions[i],
                                                         self.mask_token[self.config.model_path], text_attrs[i],
                                                         image_attrs[i], self.bert_tokenizer)
                else:
                    prompt = prompt_for_metaphor_base_llava(line, input_captions[i],
                                                      self.mask_token[self.config.model_path], cot_result[i],
                                                      image_attrs[i])
            new_tokens.append(prompt)
            # print(prompt)
        # truncation = not self.config.cot
        batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                       truncation=False,
                                                       max_length=self.config.max_length)
        batch_input = batch_input.data
        if not self.config.model_path.startswith('google'):
            for input_ids in batch_input['input_ids']:
                if 50264 not in input_ids.tolist():
                    print(input_ids, ', shape:', input_ids.shape, ', max length:', self.config.max_length)
                if self.config.special:
                    poss = []
                    for i in range(input_ids.shape[0]):
                        if (input_ids[i] == 50264):
                            poss.append(i)
                    mask_pos.append(poss)
                else:
                    mask_pos.append(input_ids.tolist().index(
                        self.special_token_mapping[self.mask_token[self.config.model_path]]))
                # print(mask_pos)
        labels = [self.config.label_list[int(w)] for w in input_labels]


        batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                truncation=False,
                                                return_tensors="pt").data
        if self.config.reasoning == 'mprompt':
            image_path = self.config.image_path
            images = []
            for i, id in enumerate(ids):
                path = os.path.join(image_path, id + '.jpg')
                # print("path:",path)
                try:
                    image = Image.open(path).convert('RGB')
                    image = self.transform(image)
                    images.append(image)
                except Exception:
                    print('no image:', path)
                    image = torch.zeros([3, 224, 224])

        res = {
            'id': ids,
            'input_ids': batch_input['input_ids'],
            'input_masks': batch_input['attention_mask'],
            'output_ids': batch_output['input_ids'],
            'output_masks': batch_output['attention_mask'],
            'input_labels': torch.tensor(input_labels),
            'mask_pos': torch.tensor(mask_pos),
            'image_prefix': torch.stack(image_prefix),
        }
        # print(torch.stack(image_blips).shape)
        for k, v in res.items():
            if k != 'target' and k != 'source' and k != 'id':
                res[k] = v.to(self.config.device)
            else:
                res[k] = v
        return res


class Preprocessor:
    def __init__(self, config):
        self.config = config

    def read_file(self):
        dataname = self.config.dataname
        if dataname == 'twitter' or dataname == 'fb':
            train_file = os.path.join(self.config.data_dir,
                                      'metaphor_{}_train.pkl'.format(
                                          dataname))
            test_file = os.path.join(self.config.data_dir,
                                     'metaphor_{}_test.pkl'.format(
                                         dataname))
            train_data = pkl.load(open(train_file, 'rb'))
            test_data = pkl.load(open(test_file, 'rb'))
            ids = np.arange(len(train_data))
            np.random.shuffle(ids)
            lens = math.ceil(len(train_data['id']) * 0.1)
            valid_data = {w: v[-lens:] for w, v in train_data.items()}
            train_data = {w: v[:-lens] for w, v in train_data.items()}
        else:
            train_file = os.path.join(self.config.data_dir,
                                      'metaphor_{}_train.pkl'.format(
                                          dataname))
            valid_file = os.path.join(self.config.data_dir,
                                      'metaphor_{}_dev.pkl'.format(
                                          dataname))
            test_file = os.path.join(self.config.data_dir,
                                     'metaphor_{}_test.pkl'.format(
                                         dataname))
            train_data = pkl.load(open(train_file, 'rb'))
            test_data = pkl.load(open(test_file, 'rb'))
            valid_data = pkl.load(open(valid_file, 'rb'))
            print('train_data:', len(train_data['raw_texts']))
        return train_data, valid_data, test_data

    def metaphor2indices(self, cur_data):
        res = []
        for i in range(len(cur_data['raw_texts'])):
            ids = cur_data['id'][i]
            text = cur_data['raw_texts'][i]
            label = cur_data['labels'][i]
            if self.config.dataname == 'meme':
                res.append([ids, text, label])
                continue
            if self.config.dataname != 'meme':
                caption = cur_data['captions'][i]
                template = cur_data['template'][i]
                text_attrs = cur_data['text_attributes'][i]
                image_attrs = cur_data['image_attributes'][i]
                target = cur_data['target'][i]
                source = cur_data['source'][i]
                image_prefix = cur_data['image_prefix'][i]
                cot_result = cur_data['cot_result'][i]
                res.append([ids, text, label, caption, template, text_attrs, image_attrs, target, source, image_prefix,
                        cot_result])
            #     res.append([ids, text, label, caption, template, text_attrs, image_attrs, target, source])
            # res.append([ids, text, label, caption, template, text_attrs, image_attrs, target, source])
            else:
                res.append([ids, text, label, caption, [], [], [], [], [], image_prefix])
        return res

    def forward(self):
        modes = 'train valid test'.split()
        dataset = self.read_file()
        res = []
        for i, mode in enumerate(modes):
            data = self.metaphor2indices(dataset[i])
            res.append(data)
        return res
