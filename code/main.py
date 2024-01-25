import argparse
import yaml
import torch
from attrdict import AttrDict
import pandas as pd
from torch import nn
from transformers import AutoConfig

from src.utils import set_seed, load_params_LLM, frozen
from src.loader import MetaphorDataLoader, MetaphorLLaVADataLoader
from src.model import MMetaphorCOTModel, ExtractImagePrefix, MetaphorLLaVABackBone
from src.engine import ImageTrainer, MMetaphorCOTTrainer, MetaphorLLaVATrainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


class Template:
    def __init__(self, args):
        config = AttrDict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        names = []
        for k, v in vars(args).items():
            setattr(config, k, v)
        config.dataname = config.data_name
        set_seed(config.seed)

        config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        names = [config.model_size, config.dataname] + names
        config.save_name = '_'.join(list(map(str, names))) + '_{}_{}_{}.pth.tar'
        self.config = config

    def forward(self):
        print(f"Running on the {self.config.data_name} data.")
        if self.config.use_llava:
            (self.trainLoader, self.validLoader, self.testLoader), self.config = MetaphorLLaVADataLoader(
                self.config).get_data()
            self.model = MetaphorLLaVABackBone(config=self.config).to(self.config.device)
            self.model = frozen(self.model)
            self.config = load_params_LLM(self.config, self.model, self.trainLoader)

            trainer = MetaphorLLaVATrainer(self.model, self.config, self.trainLoader, self.validLoader,
                                           self.testLoader)
            print("Choosing llava prompt mode.")
        elif self.config.reasoning == 'mprompt':
            (self.trainLoader, self.validLoader, self.testLoader), self.config = MetaphorDataLoader(
                self.config).get_data()
            self.model = MMetaphorCOTModel(config=self.config).to(self.config.device)
            if self.config.multi_gpu:
                self.model = nn.DataParallel(self.model)
                self.model = self.model.cuda()
            self.model = frozen(self.model)

            self.config = load_params_LLM(self.config, self.model, self.trainLoader)
            trainer = MMetaphorCOTTrainer(self.model, self.config, self.trainLoader, self.validLoader,
                                          self.testLoader)
            print("Choosing multimodal prompt mode.")

        elif self.config.reasoning == 'extract':
            (self.trainLoader, self.validLoader, self.testLoader), self.config = MetaphorDataLoader(
                self.config).get_data()
            self.model = ExtractImagePrefix(config=self.config).to(self.config.device)
            self.config = load_params_LLM(self.config, self.model, self.trainLoader)
            print("Choosing multimodal prompt mode.")
            trainer = ImageTrainer(self.model, self.config, self.trainLoader, self.validLoader,
                                   self.testLoader)
        else:
            raise 'Should choose a correct reasoning mode'

        if self.config.zero_shot == True:
            print("Zero-shot mode for evaluation.")
            r = trainer.evaluate_step(self.testLoader, 'test')
            print(r)
            return

        if self.config.reasoning == 'extract':
            print("Extract image feature.")
            test_ = trainer.evaluate_step(self.testLoader, 'test')
            dev_ = trainer.evaluate_step(self.validLoader, 'dev')
            train_ = trainer.evaluate_step(self.trainLoader, 'train')
            torch.save(train_, './data/preprocessed/image_meme_prefix_train.pt')
            torch.save(test_, './data/preprocessed/image_meme_prefix_test.pt')
            torch.save(dev_, './data/preprocessed/image_meme_prefix_dev.pt')
            return

        if self.config.inferrence:
            res = trainer.inferrence(
                    '/aworkspace/data/save/base_meme_2_2023-12-07-15-43-07.pth.tar')
            print(res)
            return
        trainer.train()
        lines = trainer.lines

        df = pd.DataFrame(lines)
        print(df.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda_index', default=1)
    parser.add_argument('-r', '--reasoning', default='mprompt',
                        choices=['mprompt', 'extract'],
                        help='multimodal prompt or extract image feature')
    parser.add_argument('-z', '--zero_shot', action='store_true', default=False,
                        help='running under zero-shot mode or fine-tune mode')
    parser.add_argument('-d', '--data_name', default='laptops',
                        choices=['met', 'meme'],
                        help='semeval data name')
    parser.add_argument('-f', '--config', default='./config/config.yaml', help='config file')
    parser.add_argument('-m', '--multi_gpu', action='store_true', default=False, help='use multi gpu')
    parser.add_argument('-s', '--special', action='store_true', default=False, help='special mode')
    parser.add_argument('-us', '--use_sim', action='store_true', default=False, help='special mode')
    args = parser.parse_args()
    template = Template(args)
    template.forward()
