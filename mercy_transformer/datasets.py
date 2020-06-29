import torch

class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, text, labels, bert, max_len):

        assert len(text) == len(labels)

        self.text = text
        self.labels = labels
        self.model = bert
        self.max_len = max_len
        self.class_list = sorted(list(set(self.labels)))

    def __getitem__(self, idx):
        text = self.text[idx]
        labels = self.labels[idx]
        
        ids = self.model.tokenize(text, max_length=self.max_len)
        labels = self.class_list.index(labels)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return ids, labels

    def __len__(self):
        return len(self.text)

class PairedQuestionDataset(torch.utils.data.Dataset):

    def __init__(self, question1, question2, labels, bert, max_len):

        assert len(question1) == len(question2)
        assert len(labels) == len(question1)

        self.question1 = question1
        self.question2 = question2
        self.labels = labels
        self.model = bert
        self.max_len = max_len
        self.class_list = sorted(list(set(self.labels)))

    def __getitem__(self, idx):
        question1 = self.question1[idx]
        question2 = self.question2[idx]
        labels = self.labels[idx]

        ids1 = self.model.tokenize(
            question1, max_length=self.max_len)
        ids2 = self.model.tokenize(
            question2, max_length=self.max_len)
        labels = self.class_list.index(labels)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return ids1, ids2, labels

    def __len__(self):
        return len(self.question1)