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