import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter, defaultdict

from nltk.tokenize import WordPunctTokenizer

import vars

def text_tokenize(text: str) -> list[str]:
    tokenizer = WordPunctTokenizer()
    text = tokenizer.tokenize(text.lower())
    
    return text

class NGram:
    def  __init__(self, train_text: list[str], n_gram: int=3, delta: float=0.1, alfas: np.array=np.array([0.3, 0.7])):
        self.train_text = train_text
        self._vocabulary = None
        self.counter_seq = defaultdict(int)
        self.n_gram = n_gram
        self.delta = delta
        self.alfas = alfas

    @property
    def vocabulary(self) -> list[str]:
        if not self._vocabulary:
            voc = list(set(self.train_text))
            # self._vocabulary = pd.Series(index=voc, data=range(len(voc)), name="vocabulary")
            self._vocabulary = voc
        return self._vocabulary

    def train(self):
        for i in range(len(self.train_text)):
            # word = self.train_text[i]
            for j in range(self.n_gram):
                ngram = self.train_text[max(i-j, 0):i+1]
                # new_ngram = ngram[max(i-j, 0):i+1]
                context_seq = " ".join(ngram)
                self.counter_seq[context_seq] += 1

    def get_probs(self, context: list[str]):
        if len(context) == 0:
            return self.get_probs_uni()
        voc_len = len(self.vocabulary)
        probs = np.zeros(voc_len)
        for i in range(voc_len):
            word = self.vocabulary[i]
            ngram = context + [word]

            context_seq = " ".join(context)
            ngram_seq = " ".join(ngram)

            posterior = self.counter_seq.get(ngram_seq, 0) + self.delta
            prior = self.counter_seq.get(context_seq, 0) + self.delta * voc_len

            probs[i] = posterior / prior

        return probs
    
    def get_probs_uni(self):
        voc_len = len(self.vocabulary)
        probs = np.zeros(voc_len)
        all_count = sum(list(self.counter_seq.values()))

        for i in range(voc_len):
            word = self.vocabulary[i]
            probs[i] = self.counter_seq.get(word, 0) / all_count

        return probs
    
    def sample_top_p(self, probs, p=0.8) -> str:
        probs = np.asarray(probs)

        # сортировка по убыванию
        idx = np.argsort(probs)[::-1]
        sorted_probs = probs[idx]

        # накопленная сумма
        cum_probs = np.cumsum(sorted_probs)

        # берем минимальный набор с суммой >= p
        cut = np.searchsorted(cum_probs, p) + 1
        idx = idx[:cut]

        # перенормировка
        norm_probs = probs[idx]
        norm_probs = norm_probs / norm_probs.sum()

        # семплирование
        sampled_i = np.random.choice(idx, p=norm_probs)
        return self.vocabulary[sampled_i]


    def predict_next(self, text: str, sample: str="top_p", p: float=0.8) -> tuple[str, np.array]:
        context = text[max(len(text)-self.n_gram+1, 0):len(text)]
        context_seq = " ".join(context)

        while not self.counter_seq.get(context_seq) and len(context) > 1:
            context = context[1:]
            context_seq = " ".join(context)

        if not self.counter_seq.get(context_seq):
            probs = self.get_probs_uni()
        else:
            if len(context) == self.n_gram-1:
                probs = self.get_probs(context)
            else:
                probs = np.zeros(len(self.vocabulary))
                probs += self.get_probs(context) * self.alfas[len(context)-1]
                while len(context) > 0:
                    context = context[1:]
                    probs += self.get_probs(context) * self.alfas[len(context)-1]

        if sample == "top_p":
            pred_token = self.sample_top_p(probs, p)
        else:
            next_i = probs.argmax()
            pred_token = self.vocabulary[next_i]

        return pred_token, probs

def clear_console():
    print("\033[2J\033[H", end="")

if __name__ == "__main__":
    with open(os.path.join(vars.DATASETS_ROOT, "quora.txt"), "r", encoding="UTF-8") as f:
        text = text_tokenize(f.read())

    # print(text[50])
        
    model = NGram(
        train_text=text,
        n_gram=3,
        delta=0.1,
        alfas=np.array([0.3, 0.7])
    )

    model.train()

    print(f"Длина словаря: {len(model.vocabulary)}")
    print(f"Кол-во ngram: {len(model.counter_seq)}")
    # print(model.counter_seq)

    val_text = "How i"
    val_text_tokenized = text_tokenize(val_text)

    clear_console()
    print(" ".join(val_text_tokenized))

    while len(val_text_tokenized) < 100:
        pred_token, probs = model.predict_next(val_text_tokenized)

        pairs = list(zip(model.vocabulary, probs))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

        val_text_tokenized.append(pred_token)

        # os.system('cls' if os.name == 'nt' else 'clear')
        clear_console()
        print(" ".join(val_text_tokenized))
        # print(pairs_sorted[:5])
        # print(pairs_sorted[-5:])




    