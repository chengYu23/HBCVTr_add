import deepsmiles
from SmilesPE.tokenizer import *
from SmilesPE.pretokenizer import atomwise_tokenizer
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer
import codecs


class CustomBart_FG_Tokenizer(PreTrainedTokenizer):
    def __init__(self, vocab, *args, **kwargs):
        self.vocab = vocab  # 提前设置 vocab
        self.pad_token = "[PAD]"  # 确保 pad_token 存在于 vocab 中
        self.converter = deepsmiles.Converter(rings=True, branches=True)
        self.encoder = {char: idx for idx, char in enumerate(vocab)}
        self.decoder = {idx: char for idx, char in enumerate(vocab)}
        self.spe_vob = codecs.open('data/spe_vocab_list.txt')

        super().__init__(*args, **kwargs)

    def get_vocab(self):
        return self.vocab

    def tokenize(self, text, **kwargs):
        spe = SPE_Tokenizer(self.spe_vob)
        tokens = spe.tokenize(text)
        tokens_list = tokens.split(' ')
        return tokens_list

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
