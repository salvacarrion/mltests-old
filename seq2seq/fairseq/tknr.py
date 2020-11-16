from sacremoses import MosesTokenizer
from nltk.tokenize import word_tokenize
import fastBPE

from tokenizers.normalizers import NFD, Lowercase, Strip, StripAccents
from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
from tokenizers.pre_tokenizers import Whitespace


class FairTokenizer:
    def __init__(self, lang, tokenizer="moses", bpe=None):
        super().__init__()

        # Tokenizers => nltk, space, moses
        # BPE => gpt2, bytes, sentencepiece, subword_nmt, byte_bpe, characters, bert, fastbpe, hf_byte_bpe

        self.lang = lang
        self.tokenizer = tokenizer
        self.bpe = tokenizer

        self.normalizer = normalizers.Sequence([NFD(), Strip()])  # StripAccents requires NFD  // Lowercase(), StripAccents()

        # Tokenizers
        if tokenizer == "space":
            self.tokenizer =  pre_tokenizers.Sequence([Whitespace()])
        elif tokenizer == "moses":
            self.tokenizer_engine = MosesTokenizer(self.lang)
        elif tokenizer == "nltk":
            self.tokenizer_engine = word_tokenize
        else:
            raise ValueError("Unknown bpe engine")

        # BPE
        if bpe is None:
            pass
        elif bpe == "fastbpe":
            self.bpe_engine = fastBPE.fastBPE()
        else:
            raise ValueError("Unknown tokenizer")

    def normalize(self, x):
        return self.normalizer.normalize_str(x)

    def tokenize(self, x):
        if self.tokenizer == "space":
            return self.tokenizer.pre_tokenize_str(x)
        elif self.tokenizer == "moses":
            return self.tokenizer_engine.tokenize(x)
        elif self.tokenizer == "nltk":
            return self.tokenizer_engine(x, language=self.lang)
        else:
            raise ValueError("Unknown tokenizer")

    def apply_bpe(self, x):
        if self.bpe == "fastbpe":
            return self.bpe_engine.apply(x)
        else:
            raise ValueError("Unknown bpe engine")
