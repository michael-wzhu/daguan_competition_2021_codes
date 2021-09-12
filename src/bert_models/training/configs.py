# -*- coding: utf-8 -*-
from transformers import BertConfig, BertTokenizer

from src.bert_models.models import ClsBERT
from transformers import BertConfig, BertTokenizer

from src.bert_models.models import ClsBERT
from src.bert_models.models.modeling_bert_pabee import ClsBERTWithPABEE
from src.bert_models.models.modeling_nezha import ClsNezha

MODEL_CLASSES = {
    'bert': (BertConfig, ClsBERT, BertTokenizer),
    'bert_pabee': (BertConfig, ClsBERTWithPABEE, BertTokenizer),
    'nezha': (BertConfig, ClsNezha, BertTokenizer),
}

MODEL_PATH_MAP = {
    # 'bert': './bert_finetune_cls/resources/bert_base_uncased',
    'bert': './resources/bert/chinese-bert-wwm-ext_embedding_replaced_random',
    'albert': './resources/albert/albert_base_zh/albert_base_zh_embedding_replaced_random',
}