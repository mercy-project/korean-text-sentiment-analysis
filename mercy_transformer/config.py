from transformers import *

MODELS = {
    'bert': [BertModel, BertTokenizer, 'bert-base-multilingual-uncased'],
    'distilbert': [DistilBertModel, DistilBertTokenizer, 'distilbert-base-multilingual-cased']}