import torch.nn as nn
import torch
import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration, RobertaForMaskedLM, BertForMaskedLM, AutoModel, \
    AutoImageProcessor, AutoConfig, BertModel, AutoModelForSequenceClassification, BertLayer, AutoModelForImageClassification
from torchtext.vocab import GloVe
from transformers import T5Config, T5ForConditionalGeneration, T5EncoderModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from torch.utils.checkpoint import checkpoint
from torchvision import models


class MultimodalEncoder(nn.Module):
    def __init__(self, config):
        super(MultimodalEncoder, self).__init__()
        self.text_encoder = T5EncoderModel.from_pretrained(config.model_path)
        text_encoder_config = AutoConfig.from_pretrained(config.model_path)
        config.hidden_size = text_encoder_config.hidden_size
        self.config = config
        self.prefix_length = config.prefix_length

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            image_ids=None,
    ):
        text_features = self.text_encoder(input_ids, attention_mask).last_hidden_state
        return text_features


class MMetaphorCOTModel(nn.Module):
    def __init__(self, config):
        super(MMetaphorCOTModel, self).__init__()
        self.config = config
        self.mm_encoder = MultimodalEncoder(config)
        self.llm = T5ForConditionalGeneration.from_pretrained(config.model_path)
        self.dense_layer = nn.Sequential(
            nn.Linear(self.config.vision_dim, self.config.hidden_size * self.config.prefix_length),
            # nn.BatchNorm1d(self.config.hidden_size * self.config.prefix_length),
            nn.Dropout(self.config.dropout),
            # nn.Linear((self.config.hidden_size * self.config.prefix_length) * 2,
            #           self.config.hidden_size * self.config.prefix_length),
            nn.ReLU(inplace=True)
        )

        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size,
                                                     vdim=config.hidden_size, num_heads=4, batch_first=True)

        self.gate_dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    def forward(
            self,
            **kwargs
    ):
        # input_ids, input_masks, image_ids, output_ids, output_masks = [kwargs[w] for w in '\
        #                             input_ids, input_masks, image_ids, output_ids, output_masks'.strip().split(
        #     ', ')]
        input_ids, input_masks, image_prefix, output_ids, output_masks = [kwargs[w] for w in '\
                                            input_ids, input_masks, image_prefix, output_ids, output_masks'.strip().split(
            ', ')]
        output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100
        text_features = self.mm_encoder(input_ids, input_masks, image_prefix)
        vision_features = self.dense_layer(image_prefix)
        vision_features = vision_features.view(-1, self.config.prefix_length, self.config.hidden_size)
        vision_features, _ = self.mha_layer(text_features, vision_features, vision_features)
        merge = torch.cat([text_features, vision_features], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        embedded = (1 - gate) * text_features + gate * vision_features
        output = self.llm(inputs_embeds=embedded, decoder_input_ids=None,
                          decoder_attention_mask=output_masks, labels=output_ids)
        loss = output[0]
        return loss

    def forward_entity(self, embedded, output_ids, output_masks, *args):
        output = self.llm(inputs_embeds=embedded, decoder_input_ids=None,
                          decoder_attention_mask=output_masks, labels=output_ids)
        return output[0]

    def generate(self, **kwargs):
        input_ids, input_masks, image_prefix = [kwargs[w] for w in '\
                                            input_ids, input_masks, image_prefix'.strip().split(
            ', ')]
        # input_ids, input_masks, image_ids = [kwargs[w] for w in '\
        #                                     input_ids, input_masks, image_ids'.strip().split(
        #     ', ')]
        text_features = self.mm_encoder(input_ids, input_masks, image_prefix)

        # vision_features = self.dense_layer(vision_features)
        vision_features = self.dense_layer(image_prefix)
        vision_features = vision_features.view(-1, self.config.prefix_length, self.config.hidden_size)
        vision_features, _ = self.mha_layer(text_features, vision_features, vision_features)
        # print("vision:", vision_features.shape)
        # print("text:", text_features.shape)
        merge = torch.cat([text_features, vision_features], dim=-1)
        # print("merge:", merge.shape)
        gate = self.sigmoid(self.gate_dense(merge))
        embedded = (1 - gate) * text_features + gate * vision_features
        output = self.llm.generate(inputs_embeds=embedded, max_length=self.config.max_length)
        dec = [self.tokenizer.decode(ids) for ids in output]
        output = [context.replace('<pad>', '').replace('</s>', '').replace('<extra_id_0>', '').replace('<extra_id_1>',
                                                                                                       '').replace(
            '<extra_id_2>', '').strip() for context in
                  dec]
        return output

    def evaluate(self, step_one=False, **kwargs):
        input_ids, input_masks, image_prefix = [kwargs[w] for w in '\
                                                    input_ids, input_masks, image_prefix'.strip().split(
            ', ')]
        # input_ids, input_masks, image_ids = [kwargs[w] for w in '\
        #                                             input_ids, input_masks, image_ids'.strip().split(
        #     ', ')]
        text_features = self.mm_encoder(input_ids, input_masks, image_prefix)
        vision_features = self.dense_layer(image_prefix)
        vision_features = vision_features.view(-1, self.config.prefix_length, self.config.hidden_size)
        vision_features, _ = self.mha_layer(text_features, vision_features, vision_features)
        merge = torch.cat([text_features, vision_features], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        embedded = (1 - gate) * text_features + gate * vision_features
        output = self.llm.generate(inputs_embeds=embedded, max_length=self.config.max_length)
        dec = [self.tokenizer.decode(ids) for ids in output]
        # print(dec)
        label_dict = {w: i for i, w in enumerate(self.config.label_list)}
        output = [label_dict.get(
            w.replace('<pad>', '').replace('</s>', '').replace('<extra_id_0>', '').replace('<extra_id_1>', '').replace(
                '<extra_id_2>', '').strip(), 0) for w in dec]
        # print(output)
        return output


class ExtractImagePrefix(nn.Module):
    def __init__(self, config):
        super(ExtractImagePrefix, self).__init__()
        self.vision_encoder = models.resnet50(pretrained=True)

    def forward(
            self,
            **kwargs
    ):
        vision_features = self.vision_encoder(kwargs['image_ids'])
        return vision_features


class MetaphorLLaVABackBone(nn.Module):
    def __init__(self, config):
        super(MetaphorLLaVABackBone, self).__init__()
        self.config = config
        # self.engine = T5ForConditionalGeneration.from_pretrained(config.model_path)
        self.model = self.getEngine(config.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.loss = nn.CrossEntropyLoss()
        self.mm_encoder = MultimodalEncoder(config)
        self.dense_layer = nn.Sequential(
            nn.Linear(self.config.vision_dim, self.config.hidden_size * self.config.prefix_length),
            # nn.Linear(self.config.vision_dim, self.config.hidden_size),
            # nn.BatchNorm1d(self.config.hidden_size * self.config.prefix_length),
            nn.Dropout(self.config.dropout),
            # nn.Linear((self.config.hidden_size * self.config.prefix_length) * 2,
            #           self.config.hidden_size * self.config.prefix_length),
            nn.ReLU(inplace=True)
        )
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size,
                                                     vdim=config.hidden_size, num_heads=self.config.num_heads,
                                                     batch_first=True)
        self.gate_dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, **kwargs):
        input_ids, input_masks, output_ids, output_masks, input_labels, image_prefix, mask_pos = [kwargs[w] for w in '\
                input_ids, input_masks, output_ids, output_masks, input_labels, image_prefix, mask_pos'.strip().split(
            ', ')]
        # input_ids, input_masks, output_ids, output_masks, input_labels, image_blip, mask_pos = [kwargs[w] for w in '\
        #         input_ids, input_masks, output_ids, output_masks, input_labels, image_blip, mask_pos'.strip().split(
        #     ', ')]
        # 由self.model计算出loss返回给模型
        output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100
        text_features = self.mm_encoder(input_ids, input_masks, image_prefix)
        vision_features = self.dense_layer(image_prefix)
        vision_features = vision_features.view(-1, self.config.prefix_length, self.config.hidden_size)
        vision_features, _ = self.mha_layer(text_features, vision_features, vision_features)
        merge = torch.cat([text_features, vision_features], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        embedded = (1 - gate) * text_features + gate * vision_features
        output = self.model(embedded, input_labels, output_ids, output_masks, mask_pos)

        return output

    def generate(self, **kwargs):
        input_ids, input_masks, image_prefix = [kwargs[w] for w in '\
                        input_ids, input_masks, image_prefix'.strip().split(
            ', ')]
        text_features = self.mm_encoder(input_ids, input_masks, image_prefix)
        vision_features = self.dense_layer(image_prefix)
        vision_features = vision_features.view(-1, self.config.prefix_length, self.config.hidden_size)
        vision_features, _ = self.mha_layer(text_features, vision_features, vision_features)
        merge = torch.cat([text_features, vision_features], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        embedded = (1 - gate) * text_features + gate * vision_features
        output = self.model.generate(embedded)
        return output

    def evaluate(self, **kwargs):
        input_ids, input_masks, output_ids, output_masks, input_labels, image_prefix, mask_pos = [
            kwargs[w] for w in '\
                            input_ids, input_masks, output_ids, output_masks, input_labels, image_prefix, mask_pos'.strip().split(
                ', ')]
        text_features = self.mm_encoder(input_ids, input_masks, image_prefix)
        vision_features = self.dense_layer(image_prefix)
        vision_features = vision_features.view(-1, self.config.prefix_length, self.config.hidden_size)
        vision_features, _ = self.mha_layer(text_features, vision_features, vision_features)
        merge = torch.cat([text_features, vision_features], dim=-1)
        gate = self.sigmoid(self.gate_dense(merge))
        embedded = (1 - gate) * text_features + gate * vision_features
        output = self.model.evaluate(embedded, mask_pos)
        return output

    def getEngine(self, model_name):
        if model_name == 'google/flan-t5-base' or model_name == 'google/flan-t5-large':
            print('use T5ForConditionalGeneration model:{}'.format(model_name))
            return MetaphorLLaVAT5Base(self.config)
        elif model_name == 'bert-base-uncased' or model_name == 'bert-large-uncased':
            print('use BertForMaskedLM model:{}'.format(model_name))


class MetaphorLLaVAT5Base(nn.Module):
    def __init__(self, config):
        super(MetaphorLLaVAT5Base, self).__init__()
        self.config = config
        self.engine = T5ForConditionalGeneration.from_pretrained(config.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    def forward(self, input_embedded, input_labels, output_ids, output_masks, mask_pos):
        # print(output_ids)
        # output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100
        output = self.engine(inputs_embeds=input_embedded, decoder_input_ids=None,
                             decoder_attention_mask=output_masks, labels=output_ids)
        # print(output)
        loss = output[0]
        return loss

    def generate(self, input_embedded):
        output = self.engine.generate(inputs_embeds=input_embedded, max_length=50)
        dec = [self.tokenizer.decode(ids) for ids in output]
        output = [context.replace('<pad>', '').replace('</s>', '').replace('<extra_id_0>', '').strip() for context in
                  dec]
        return output

    def evaluate(self, input_embedded, mask_pos):
        output = self.engine.generate(inputs_embeds=input_embedded, max_length=50)
        dec = [self.tokenizer.decode(ids) for ids in output]
        # print(dec)
        label_dict = {w: i for i, w in enumerate(self.config.label_list)}
        output = [label_dict.get(w.replace('<pad>', '').replace('</s>', '').strip(), 0) for w in dec]
        # print(output)
        return output