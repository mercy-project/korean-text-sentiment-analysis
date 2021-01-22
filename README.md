# korean-text-sentiment-analysis

이 저장소는 [mercy-project](https://github.com/mercy-project)의 한 부분으로써 한국어를 이용한 딥러닝에 사용되는 것들을 다룹니다.

## Introduction

- 이 프로젝트의 목적은 huggingface의 transformers 저장소를 사용하기 편하도록 wrapping하는 것입니다.
- 또한 Pretrained Language Model(from huggingface)을 사용하여 간단하게 비슷한 의미를 가지는 문장을 찾을 수 있는 metric을 제공합니다.

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

sudo python3 -m pip install .

pip3 install -r requirements.txt
```

## Quick start

- how to get similar text with latent vector

```python
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
```

To Run

```
python3 smilar.py
```

Result

```
[[4.17170231]
 [6.63776399]
 [4.8249083 ]
 [5.71683576]]
[[0.07849896]
 [0.20911893]
 [0.11204922]
 [0.15967475]]
```

- classfication

```python
from mercy_transformer import models
from mercy_transformer import metric
from mercy_transformer import datasets

import torch
import torch.nn as nn

class Classifier(nn.Module):

    def __init__(self, bert, num_class):
        super(Classifier, self).__init__()

        self.bert = bert
        self.classifier = nn.Linear(768, num_class)

    def forward(self, ids):
        latent = self.bert(ids)
        latent = latent[:, 0]
        logits = self.classifier(latent)
        return logits

bert = models.LanguageModel('distilbert')
model = Classifier(
    bert=bert,
    num_class=2)

classfication_datasets = datasets.ClassificationDataset(
    text=['아 더빙.. 진짜 짜증나네요 목소리',
          '흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나',
          '너무재밓었다그래서보는것을추천한다',
          '교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정',
          '사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다',
          '막 걸음마 뗀 3세부터 초등학교 1학년생인 8살용영화.ㅋㅋㅋ...별반개도 아까움.',
          '원작의 긴장감을 제대로 살려내지못했다.',
          '별 반개도 아깝다 욕나온다 이응경 길용우 연기생활이몇년인지..정말 발로해도 그것보단 낫겟다 납치.감금만반복반복..이드라마는 가족도없다 연기못하는사람만모엿네',
          '액션이 없는데도 재미 있는 몇안되는 영화'],
    labels=[0, 1, 0, 0, 1, 0, 0, 0, 1],
    bert=bert,
    max_len=30)

train_loader = torch.utils.data.DataLoader(
    dataset=classfication_datasets,
    batch_size=32,
    num_workers=1)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=1e-4)

for epoch in range(10):

    for step, (ids, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        logits = model(ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        pred = torch.argmax(logits, axis=1)
        acc = pred.eq(labels).sum().item() / ids.shape[0]

        print(epoch, step, loss.item(), acc)
```

To Run

```
python3 classfication.py
```

Result

```
0 0 0.737722635269165 0.3333333333333333
1 0 0.6059796810150146 0.6666666666666666
2 0 0.4729032516479492 0.7777777777777778
3 0 0.3866463899612427 0.8888888888888888
4 0 0.24941475689411163 1.0
5 0 0.1359175443649292 1.0
6 0 0.06440091878175735 1.0
7 0 0.027132326737046242 1.0
8 0 0.008938517421483994 1.0
9 0 0.0025468349922448397 1.0
```

- paired question

```python
from mercy_transformer import models
from mercy_transformer import metric
from mercy_transformer import datasets

import torch
import torch.nn as nn

class PairedQuestion(nn.Module):

    def __init__(self, bert):
        super(PairedQuestion, self).__init__()

        self.bert = bert
        self.classifier = nn.Linear(768 * 2, 2)

    def forward(self, ids1, ids2):
        latent1 = self.bert(ids1)[:, 0]
        latent2 = self.bert(ids2)[:, 0]
        concat = torch.cat([latent1, latent2], axis=1)
        logits = self.classifier(concat)
        return logits


bert = models.LanguageModel('distilbert')
model = PairedQuestion(
    bert=bert)

paired_dataset = datasets.PairedQuestionDataset(
    question1=['골프 배워야 돼',
               '많이 늦은시간인데 연락해봐도 괜찮을까?',
               '물배달 시켜야겠다.',
               '배고파 죽을 것 같아',
               '심심해',
               '나 그 사람이 좋아'],
    question2=['골프치러 가야돼',
               '늦은 시간인데 연락해도 괜찮을까?',
               '물 주문해야지',
               '배 터질 것 같아',
               '방학동안 너무 즐거웠어',
               '너무 싫어'],
    labels=['sim', 'sim', 'sim', 'unsim', 'unsim', 'unsim'],
    bert=bert,
    max_len=40)

train_loader = torch.utils.data.DataLoader(
    dataset=paired_dataset,
    batch_size=32,
    num_workers=2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=1e-4)

for epoch in range(20):

    for step, (ids1, ids2, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        logits = model(ids1, ids2)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        pred = torch.argmax(logits, axis=1)
        acc = pred.eq(labels).sum().item() / ids1.shape[0]

        print(epoch, step, loss.item(), acc)
```

To Run

```
python3 paired_question.py
```

Result

```
0 0 0.7150420546531677 0.5
1 0 0.8219130039215088 0.5
2 0 0.8126139640808105 0.5
3 0 0.6669130325317383 0.6666666666666666
4 0 0.6324862837791443 0.6666666666666666
5 0 0.5813004970550537 1.0
6 0 0.45990419387817383 1.0
7 0 0.28030848503112793 1.0
8 0 0.1331692934036255 1.0
9 0 0.06911587715148926 1.0
10 0 0.030566172674298286 1.0
11 0 0.015069677494466305 1.0
12 0 0.008203997276723385 1.0
13 0 0.004712474066764116 1.0
14 0 0.0029395369347184896 1.0
15 0 0.0020999612752348185 1.0
16 0 0.001615511137060821 1.0
17 0 0.0012224833481013775 1.0
18 0 0.0008657379657961428 1.0
19 0 0.0006126383086666465 1.0
```

## Todo List

- [ ] GPU Assign
- [x] Classification
- [x] Paired Question
