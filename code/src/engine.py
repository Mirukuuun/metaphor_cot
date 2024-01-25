import os
import pickle
import time
import logging

import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict

from src.utils import prompt_for_connection, prompt_for_metaphor_label, log_config, get_parameter_number


class MMetaphorCOTTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader) -> None:
        self.model = model
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        self.final_score = 0
        self.final_res = ''
        self.scores, self.lines = [], []
        log_name = os.path.join(config.target_dir,
                                'COT_log' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) + '.log')
        logging.basicConfig(filename=log_name, filemode="w", format="\n%(levelname)s:%(message)s\n",
                            datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
        self.logger = logging
        log_config(self.config, self.logger)
        self.re_init()

    def train(self):
        best_score, best_iter = 0, -1
        self.logger.info("START TRAINING...")
        print(self.config.optimizer.state_dict()['param_groups'][0]['lr'])
        parameters_num = get_parameter_number(self.model)
        print("parameters num:{}, trainable parameters num:{}".format(parameters_num['Total'],
                                                                      parameters_num['Trainable']))
        for epoch in tqdm(range(self.config.epoch_size)):
            self.logger.info(f"TRAIN EPOCH: {epoch}")
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            self.train_step()
            result = self.evaluate_step(mode='valid')
            self.logger.info('Epoch {}, Valid F1: {}, ACC: {}'.format(self.global_epoch, result['F1'], result['Acc']))
            print('Epoch {}, Valid F1: {}, ACC: {}'.format(self.global_epoch, result['F1'], result['Acc']))
            self.re_init()
            score = result['default']
            self.add_instance(result)
            res = self.get_best()

            if score > best_score:
                best_score, best_iter = score, epoch
                save_name = self.save_name.format(epoch, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
                if not os.path.exists(self.config.target_dir):
                    os.makedirs(self.config.target_dir)
                torch.save({'epoch': epoch, 'model': self.model.cpu().state_dict()},
                           save_name)
                self.logger.info("***EPOCH: {} , best score update: {}".format(epoch, best_score))
                self.model.to(self.config.device)
            elif epoch - best_iter > self.config.patience:
                self.logger.info("***Not upgrade for {} steps, early stopping...".format(self.config.patience))
                # print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break
            self.model.to(self.config.device)

        # res = self.final_evaluate(best_score, best_iter)
        score = res['default']
        self.add_instance(res)
        # save_name = self.save_name.format(epoch)
        self.final_score, self.final_res = score, res

    def prepare_step_two(self, outputs, data):
        tokenizer = self.model.module.tokenizer if self.config.multi_gpu else self.model.tokenizer
        context_ids = data['context_ids']
        output_ids, output_masks = [data[w] for w in 'output_ids, output_masks'.strip().split(', ')]
        contexts = [tokenizer.decode(ids) for ids in context_ids]
        contexts = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts]

        new_prompts = []
        new_contexts = []
        for context, output in zip(contexts, outputs):
            # prompt = prompt_for_metaphor_label2(context, output)
            new_context, prompt = prompt_for_connection(context, output)
            new_prompts.append(prompt)
            new_contexts.append(new_context)
        batch_inputs = tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                   max_length=self.config.max_length)
        batch_inputs = batch_inputs.data
        batch_contexts = self.model.tokenizer.batch_encode_plus(new_contexts, padding=True, return_tensors='pt',
                                                                max_length=self.config.max_length)
        batch_contexts = batch_contexts.data
        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'output_ids': output_ids,
            'output_masks': output_masks,
            'context_ids': batch_contexts['input_ids'],
            'image_prefix': data['image_prefix'],
            # 'image_ids': data['image_ids'],
        }

        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    def prepare_step_three(self, outputs, pre_data, data):
        tokenizer = self.model.module.tokenizer if self.config.multi_gpu else self.model.tokenizer
        context_ids = pre_data['context_ids']
        output_ids, output_masks = [data[w] for w in 'output_ids, output_masks'.strip().split(', ')]
        contexts = [tokenizer.decode(ids) for ids in context_ids]
        contexts = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts]

        new_prompts = []
        new_contexts = []
        for context, output in zip(contexts, outputs):
            prompt = prompt_for_metaphor_label(context, output)
            new_prompts.append(prompt)

        batch_inputs = tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                   max_length=self.config.max_length)
        batch_inputs = batch_inputs.data

        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'output_ids': output_ids,
            'output_masks': output_masks,
            'image_prefix': data['image_prefix'],
            # 'image_ids': data['image_ids'],
        }
        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    def prepare_train_step(self, outputs, data):
        tokenizer = self.model.module.tokenizer if self.config.multi_gpu else self.model.tokenizer
        context_ids = data['context_ids']
        output_ids, output_masks = [data[w] for w in 'output_ids, output_masks'.strip().split(', ')]
        contexts = [tokenizer.decode(ids) for ids in context_ids]
        contexts = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts]
        new_prompts = []
        for context, output in zip(contexts, outputs):
            prompt = context + f'Is this sample contains metaphor? Answer: [mask]'
            new_prompts.append(prompt)
        batch_inputs = tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                   max_length=self.config.max_length)
        batch_inputs = batch_inputs.data

        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'output_ids': output_ids,
            'output_masks': output_masks,
            'image_prefix': data['image_prefix'],
            # 'image_ids': data['image_ids'],
        }
        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res


    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader, total=self.train_loader.data_length, position=0)

        losses = []
        for i, data in enumerate(train_data):

            step_one_inferred_output = self.model.module.generate(step_one=True, train=False, **data) \
                if self.config.multi_gpu \
                else self.model.generate(step_one=True, train=False, **data)
            self.logger.info(f"Step_one_inferred_output:{step_one_inferred_output}")
            # print("step one output:", step_one_inferred_output)
            step_one_inferred_data = self.prepare_step_two(step_one_inferred_output, data)
            # print("step two input:", [self.model.tokenizer.decode(ids) for ids in step_one_inferred_data['input_ids']])
            step_two_inferred_output = self.model.generate(**step_one_inferred_data)
            # self.logger.info(f"Step_two_inferred_output:{step_two_inferred_output}")
            # print("step two output:", step_two_inferred_output)
            step_two_inferred_data = self.prepare_step_three(step_two_inferred_output, step_one_inferred_data, data)
            final_data = step_two_inferred_data
            # print("step three input:", [self.model.tokenizer.decode(ids) for ids in step_two_inferred_data['input_ids']])
            loss = self.model(**final_data)
            losses.append(loss.item())
            loss.backward()
            if (i + 1) % self.config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.config.optimizer.step()
                self.config.scheduler.step()
                self.model.zero_grad()
            description = "Epoch {}, loss:{:.4f}".format(self.global_epoch, np.mean(losses))
            train_data.set_description(description)

    def evaluate_step(self, dataLoader=None, mode='valid'):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        dataiter = dataLoader
        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length, position=0):
            with torch.no_grad():
                step_one_inferred_output = self.model.module.generate(step_one=True, train=False, **data) \
                    if self.config.multi_gpu \
                    else self.model.generate(step_one=True, train=False, **data)
                self.logger.info(f"Step_one_inferred_output: {step_one_inferred_output}")
                step_one_inferred_data = self.prepare_step_two(step_one_inferred_output, data)
                step_two_inferred_output = self.model.generate(**step_one_inferred_data)
                # self.logger.info(f"Step_two_inferred_output: {step_two_inferred_output}")
                step_two_inferred_data = self.prepare_step_three(step_two_inferred_output, step_one_inferred_data, data)
            # output = self.model.evaluate(**step_two_inferred_data)
            output = self.model.evaluate(**step_two_inferred_data)
            self.add_output(data, output)

        result = self.report_score(mode=mode)
        return result

    def inferrence(self, path=''):
        self.model.load_state_dict(torch.load(path, map_location=self.config.device)['model'])
        self.model.eval()
        res = self.evaluate_step(self.test_loader, mode='test')
        self.add_instance(res)
        # self.report_score()
        return res

    def add_instance(self, res):
        self.lines.append(res)

    def get_best(self):
        best_id = np.argmax([w['default'] for w in self.lines])
        res = self.lines[best_id]
        return res

    def re_init(self):
        self.preds, self.golds = defaultdict(list), defaultdict(list)
        self.keys = ['total']

    def add_output(self, data, output):
        gold = data['input_labels']
        # print('output:', output)
        if self.config.model_path.startswith('google'):
            self.preds['total'] += output
        else:
            self.preds['total'] += output.cpu()
        self.golds['total'] += gold.tolist()
        self.golds['id'] += data['ids']

    def report_score(self, mode='valid'):
        res = {}
        res['Acc'] = accuracy_score(self.golds['total'], self.preds['total'])
        res['F1'] = f1_score(self.golds['total'], self.preds['total'], labels=[0, 1], average='macro')
        res['default'] = res['F1']
        res['mode'] = mode
        return res

class MetaphorLLaVATrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader) -> None:
        self.model = model
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        self.final_score = 0
        self.final_res = ''
        log_name = os.path.join(config.target_dir,
                                'COT_log' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) + '.log')
        logging.basicConfig(filename=log_name, filemode="w", format="\n%(levelname)s:%(message)s\n",
                            datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
        self.logger = logging
        log_config(self.config, self.logger)
        self.scores, self.lines = [], []
        self.re_init()

    def train(self):
        best_score, best_iter = 0, -1
        print("train epoch size:", self.config.epoch_size, ' lr:', self.config.bert_lr)
        parameters_num = get_parameter_number(self.model)
        print("parameters num:{}, trainable parameters num:{}".format(parameters_num['Total'],
                                                                      parameters_num['Trainable']))
        for epoch in tqdm(range(self.config.epoch_size)):
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            self.train_step()
            result = self.evaluate_step(mode='valid')
            print('Epoch {}, Valid F1: {}, ACC: {}'.format(self.global_epoch, result['F1'], result['Acc']))
            self.re_init()
            score = result['default']

            self.add_instance(result)

            res = self.get_best()

            if score > best_score:
                best_score, best_iter = score, epoch
                save_name = self.save_name.format(epoch, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
                if not os.path.exists(self.config.target_dir):
                    os.makedirs(self.config.target_dir)
                torch.save({'epoch': epoch, 'model': self.model.cpu().state_dict()},
                           save_name)
                self.model.to(self.config.device)
            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break
            self.model.to(self.config.device)

        # res = self.final_evaluate(best_iter)
        # score = res['default']

        # self.add_instance(res)

        # save_name = self.save_name.format(epoch)

        # self.final_score, self.final_res = score, res
        score = res['default']
        self.add_instance(res)
        # save_name = self.save_name.format(epoch)
        self.final_score, self.final_res = score, res

    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader, position=0)
        losses = []
        for i, data in enumerate(train_data):
            loss = self.model(**data)

            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            description = "Epoch {}, loss:{:.4f}".format(self.global_epoch, np.mean(losses))
            train_data.set_description(description)
            self.config.optimizer.step()
            self.config.scheduler.step()
            self.model.zero_grad()

    def evaluate_step(self, dataLoader=None, mode='valid'):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        dataiter = dataLoader
        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length, position=0):
            with torch.no_grad():
                output = self.model.evaluate(**data)
                self.add_output(data, output)
        result = self.report_score(mode=mode)
        return result

    def inferrence(self, path=''):
        self.model.load_state_dict(torch.load(path, map_location=self.config.device)['model'])
        self.model.eval()
        res = self.evaluate_step(self.test_loader, mode='test')
        self.add_instance(res)
        # self.report_score()
        return res

    def add_instance(self, res):
        self.lines.append(res)

    def get_best(self):
        best_id = np.argmax([w['default'] for w in self.lines])
        res = self.lines[best_id]
        return res

    def re_init(self):
        self.preds, self.golds = defaultdict(list), defaultdict(list)
        self.ids = defaultdict(list)
        self.keys = ['total']

    def add_output(self, data, output):
        gold = data['input_labels']
        # print('output:', output)
        if self.config.model_path.startswith('google'):
            self.preds['total'] += output
        else:
            self.preds['total'] += output.cpu()
        self.golds['total'] += gold.tolist()
        self.preds['id'] += data['id']

    def report_score(self, mode='valid'):
        res = {}
        res['Acc'] = accuracy_score(self.golds['total'], self.preds['total'])
        res['F1'] = f1_score(self.golds['total'], self.preds['total'], labels=[0, 1], average='macro')
        res['default'] = res['F1']
        res['mode'] = mode
        for k, v in res.items():
            if isinstance(v, float):
                res[k] = round(v * 100, 3)
        return res


class ImageTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader) -> None:
        self.model = model
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        self.final_score = 0
        self.final_res = ''

        self.scores, self.lines = [], []

    def train(self):
        for epoch in tqdm(range(self.config.epoch_size)):
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            self.train_step()
            result = self.evaluate_step(mode='valid')

    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader, ncols=10)
        losses = []
        for data in train_data:
            loss = self.model(**data)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            description = "Epoch {}, loss:{:.4f}".format(self.global_epoch, np.mean(losses))
            train_data.set_description(description)
            self.config.optimizer.step()
            self.config.scheduler.step()
            self.model.zero_grad()

    def evaluate_step(self, dataLoader=None, mode='valid'):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        dataiter = dataLoader
        ip = {}
        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length):
            # print(data)
            with torch.no_grad():
                output = self.model(**data)
                for i, id in enumerate(data['ids']):
                    ip[id] = output[i]
        if mode == 'train':
            data = pickle.load(open('./data/meme_train.pkl', 'rb'))
            name = './data/meme_train.pkl'
        elif mode == 'dev':
            data = pickle.load(open('./data/meme_dev.pkl', 'rb'))
            name = './data/meme_dev.pkl'
        else:
            data = pickle.load(open('./data/meme_test.pkl', 'rb'))
            name = './data/meme_test.pkl'
        for id in data['id']:
            print(ip[id])
            data['image_prefix'].append(ip[id])
        with open(name, 'wb') as f:
            pickle.dump(data, f)
        return
