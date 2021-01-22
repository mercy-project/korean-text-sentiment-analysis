from mercy_transformer import models
from mercy_transformer import metric

import torch
import numpy as np

model_name = ['bert', 'distilbert']
model_name = model_name[np.random.choice(len(model_name))]
model = models.LanguageModel(model_name)

text = [
    '안녕하세요 당신은 누구십니까?',
    '전화번호좀 알려주세요',
    '담당자가 누구인가요?',
    '같이 춤추실래요']

latent_list = []
for t in text:
    tokens = model.tokenize(t)
    latent = model.encode(tokens)[0][0]
    latent = torch.mean(latent, axis=0)
    latent_list.append(latent.detach().cpu().numpy())

latent_list = np.stack(latent_list)

reference = '안녕 너는 누구야?'

token = model.tokenize(reference)
latent = model.encode(token)[0][0]
latent = torch.mean(latent, axis=0)
latent = latent.detach().cpu().numpy()

distance = metric.euclidean(latent_list, [latent])
print(distance)
distance = metric.cosine(latent_list, [latent])
print(distance)