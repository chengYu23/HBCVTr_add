#!/usr/bin/env python
# coding: utf-8




import deepsmiles
from SmilesPE.tokenizer import *
from SmilesPE.pretokenizer import atomwise_tokenizer
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer
from transformers import PreTrainedTokenizer


class CustomBart_Atomic_Tokenizer(PreTrainedTokenizer):
    def __init__(self, vocab, *args, **kwargs):
        self.vocab = vocab  # 提前赋值，避免 get_vocab() 调用时报错
        self.converter = deepsmiles.Converter(rings=True, branches=True)
        self.encoder = {char: idx for idx, char in enumerate(vocab)}
        self.decoder = {idx: char for idx, char in enumerate(vocab)}
        
        super().__init__(*args, **kwargs)

    def tokenize(self, text, **kwargs):
        tokens = atomwise_tokenizer(text)
        return tokens

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.encoder.get(tokens, self.encoder.get("[UNK]"))
        else:
            return [self.encoder.get(token, self.encoder.get("[UNK]")) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.decoder.get(ids, "[UNK]")
        else:
            return [self.decoder.get(idx, "[UNK]") for idx in ids]

    @property
    def pad_token_id(self):
        return self.encoder.get(self.pad_token, self.encoder.get("[UNK]"))

    @pad_token_id.setter
    def pad_token_id(self, value):
        self.pad_token = self.convert_ids_to_tokens(value)



