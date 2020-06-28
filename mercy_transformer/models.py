from mercy_transformer import config

import torch

class LanguageModel:

    def __init__(self, model_name):

        assert model_name in config.MODELS.keys()

        self.model_config = config.MODELS[model_name]
        self.model = self.model_config[0].from_pretrained(self.model_config[2])
        self.tokenizer = self.model_config[1].from_pretrained(self.model_config[2])

    def tokenize(self, text):
        tokens = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)])
        return tokens

    def encode(self, tokens):
        latent = self.model(tokens)
        return latent