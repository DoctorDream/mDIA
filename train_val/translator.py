from transformers import MarianMTModel, MarianTokenizer
import torch
import time
from parameters import parse
args = parse()
device = torch.device(f"cuda:{args.gpu}")
cpu = torch.device("cpu")


class Translator(object):
    def __init__(self,fromLang, toLang):
        self.fromLang = fromLang
        self.toLang = toLang
        if toLang == 'ja':
            self.toLang = 'jap'
        self.model_name = f'Helsinki-NLP/opus-mt-{self.fromLang}-{self.toLang}'

        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.model.to(device)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)


    def translate(self, sentences):


        sentences = sentences
        translated = self.model.generate(**self.tokenizer(sentences, return_tensors="pt", padding=True).to(device)).to(cpu)
        torch.cuda.empty_cache()
        responds = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        return responds




