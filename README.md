# korean-text-sentiment-analysis
이 저장소는 [mercy-project](https://github.com/mercy-project)의 한 부분으로써 한국어를 이용한 딥러닝에 사용되는 것들을 다룹니다.

## Introduction
* 이 프로젝트의 목적은 huggingface의 transformers 저장소를 사용하기 편하도록 wrapping하는 것입니다.
* 또한 Pretrained Language Model(from huggingface)을 사용하여 간단하게 비슷한 의미를 가지는 문장을 찾을 수 있는 metric을 제공합니다.

## Dependency

```
scipy==1.5.0
torch==1.5.1
torchvision==0.6.1
transformers==2.11.0
```

## How to install

```
git clone https://github.com/mercy-project/korean-text-sentiment-analysis

cd korean-text-sentiment-analysis

pip install .
```

## Quick start

```python
from mercy_transformer import models
from mercy_transformer import metric

import torch
import numpy as np

model_name = ['bert', 'distilbert']
model_name = model_name[np.random.choice(len(model_name))]
model = models.LanguageModel('distilbert')

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
```

## Todo List

- [ ] GPU Assign
- [ ] Classification