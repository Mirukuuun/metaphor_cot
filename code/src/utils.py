import os
import random
import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_scheduler


def prompt_for_entity(text, caption, image_attrs):
    if isinstance(text, float) or text == 'Null':
        context = f'Image Caption: {caption}'
    else:
        tokens = text.split(' ')
        if len(tokens) > 100:
            tokens = tokens[:100]
        text = ' '.join(tokens)
        context = f'Text: {text} Image Caption: {caption}'
    prompt = context + f' Please provide entities that may contain metaphor.'
    return context, prompt

def  prompt_for_connection(context, output):
    new_context = context + f' May contain metaphor entities: {output}.'
    prompt = new_context + ' What kind of connection between these entities?'
    return new_context, prompt

def prompt_for_metaphor_label(context, output):
    prompt = context + f' Connection between the main entities is {output} Based on these contexts, is this sample contains metaphor? Return yes or no.'
    return prompt

def prompt_for_metaphor_t5base_llava(text, caption, mask_token, cot_result, image_attrs):
    tokens = text.split(' ')
    if len(tokens) > 100:
        tokens = tokens[:100]
    text = ' '.join(tokens)
    prompt = f'Text: {text} Image caption: {caption} Main entities: {cot_result[0]} Connection: {cot_result[1]}. Is this sample contains metaphor? Answer: [mask]'
    return prompt


def set_seed(seed):
    print("set seed:",seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # torch.set_deterministic(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


def load_params_LLM(config, model, fold_data):
    no_decay = ['bias', 'LayerNorm.weight', 'mm_encoder']
    named = (list(model.named_parameters()))
    no_decay = ['bias', 'LayerNorm.weight', 'mm_encoder']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in named if not any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr),
         'weight_decay': float(config.weight_decay)},
        {'params': [p for n, p in named if any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr),
         'weight_decay': 0.0},
    ]
    # print(list(model.state_dict()))

    optimizer = AdamW(optimizer_grouped_parameters, eps=float(config.adam_epsilon))
    if config.scheduler_type == 'linear':
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
        #                                         num_training_steps=config.epoch_size * fold_data.__len__())
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_proportion * config.epoch_size * fold_data.__len__(),
                                                num_training_steps=config.epoch_size * fold_data.__len__())
    elif config.scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=config.epoch_size * fold_data.__len__())
        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_proportion * config.epoch_size * fold_data.__len__(),
        #                                         num_training_steps=config.epoch_size * fold_data.__len__())
    else:
        scheduler = get_scheduler(optimizer, num_warmup_steps=config.warmup_steps,
                                                    num_training_steps=config.epoch_size * fold_data.__len__())
    config.score_manager = ScoreManager()
    config.optimizer = optimizer
    config.scheduler = scheduler
    return config


def frozen(model):
    # print(model)
    # frozen_layers = [model.MultimodalEncoder]
    for name, param in model.named_parameters():
        if 'mm_encoder' in name:
            # print("frozen layer:", name)
            param.requires_grad = False
    return model

def log_config(config, logger):
    logger.info(f"epoch:{config.epoch_size}")
    logger.info(f"batch_size:{config.batch_size}")
    logger.info(f"bert_lr:{config.bert_lr}")
    logger.info(f"prefix_length:{config.prefix_length}")
    logger.info(f"model_path:{config.model_path}")
    logger.info(f"vision_model_path:{config.vision_model_path}")
    logger.info(f"label_list:{config.label_list}")
    logger.info(f"cot:{config.cot}")
    logger.info(f"one_step:{config.one_step}")
    logger.info(f"dropout:{config.dropout}")
    logger.info(f"gradient_accumulation_steps:{config.gradient_accumulation_steps}")
    logger.info(f"seed:{config.seed}")

class ScoreManager:
    def __init__(self) -> None:
        self.score = []
        self.line = []

    def add_instance(self, score, res):
        self.score.append(score)
        self.line.append(res)

    def get_best(self):
        best_id = np.argmax(self.score)
        res = self.line[best_id]
        return res

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

