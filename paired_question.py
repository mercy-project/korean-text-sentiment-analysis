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