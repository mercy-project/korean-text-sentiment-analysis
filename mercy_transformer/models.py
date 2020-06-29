from mercy_transformer import config

import torch
import torch.nn as nn

class LanguageModel(nn.Module):

    def __init__(self, model_name):
        super(LanguageModel, self).__init__()

        assert model_name in config.MODELS.keys()

        self.model_config = config.MODELS[model_name]
        self.model = self.model_config[0].from_pretrained(self.model_config[2])
        self.tokenizer = self.model_config[1].from_pretrained(self.model_config[2])

    def forward(self, ids):
        latent = self.model(ids)
        return latent[0]

    def tokenize(self, text, max_length=None):
        if max_length is not None:
            tokens = torch.tensor(
                self.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=max_length,
                    pad_to_max_length=True))
            return tokens
        else:
            tokens = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)])
            return tokens

    def encode(self, tokens):
        latent = self.model(tokens)
        return latent