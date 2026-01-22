import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from functools import lru_cache, reduce
from collections import Counter, defaultdict

from nltk.tokenize import WordPunctTokenizer

import vars

def text_tokenize(text: str) -> list[str]:
    tokenizer = WordPunctTokenizer()
    text = tokenizer.tokenize(text.lower())
    
    return text

class NGram:
    def  __init__(
            self, 
            train_text: list[str], 
            n_gram: int=3, 
            delta: float=0.1, 
            alfas: np.array=np.array([0.3, 0.7]),
            min_c: int=3,
            min_C: int=3,
            T: float=1.0,
        ):
        # self.train_text = train_text
        # self.vocabulary = list(set(self.train_text))
        # self.vocab2id = dict(zip(self.vocabulary, range(len(self.vocabulary))))
        self.counter_seq = defaultdict(Counter)
        self.n_gram = n_gram
        self.delta = delta
        self.alfas = alfas
        self.min_c = min_c
        self.min_C = min_C
        self.T = T
        self.build_vocab_and_replace_unk(train_text)

    def get_prefix(self, context: list[str]) -> list[str]:
        prefix = context[max(len(context)-self.n_gram+1, 0):len(context)]
        n_unk = self.n_gram - 1 - len(prefix)
        prefix = [vars.UNK] * n_unk + prefix

        return prefix
    
    def build_vocab_and_replace_unk(self, text: list[str]) -> None:
        """
        1. Считает частоты слов
        2. Оставляет слова с count >= min_c
        3. Остальные заменяет на UNK
        4. Обновляет vocabulary и vocab2id
        """
        word_counts = Counter(text)

        # строим словарь
        vocab = {
            w for w, c in word_counts.items()
            if c >= self.min_c
        }
        vocab.add(vars.UNK)

        # заменяем редкие слова на UNK
        new_text = [
            w if w in vocab else vars.UNK
            for w in text
        ]

        self.train_text = new_text

        # обновляем словарь модели
        self.vocabulary = list(vocab)
        self.vocab2id = {w: i for i, w in enumerate(self.vocabulary)}


    def train(self):
        pad = [vars.UNK] * (self.n_gram - 1)
        text = pad + self.train_text

        for i in range(self.n_gram - 1, len(text)):
            word = text[i]
            context = text[i - self.n_gram + 1:i]
            # self.counter_seq[tuple(context)][word] += 1
            for j in range(self.n_gram):
                context = [vars.UNK] * j + context[j:]
                # context[j] = vars.UNK
                self.counter_seq[tuple(context)][word] += 1

        cleaned_vocab = {}

        for context, counter in self.counter_seq.items():
            if len(counter) >= self.min_C:
                cleaned_vocab[context] = counter

        self.counter_seq = cleaned_vocab

    def apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        if self.T <= 0:
            raise ValueError("Temperature must be > 0")

        if self.T == 1.0:
            return probs

        probs = np.asarray(probs)
        probs = np.maximum(probs, 1e-12)  # защита от 0

        probs_T = probs ** (1.0 / self.T)
        probs_T /= probs_T.sum()

        return probs_T


    @lru_cache(maxsize=100_000)
    def get_probs(self, context: tuple[str]):
        voc_len = len(self.vocabulary)
        counter = self.counter_seq.get(context, Counter())
        total = counter.total() + self.delta * voc_len
        probs = np.full(voc_len, self.delta / total)

        for word, cnt in counter.items():
            i = self.vocab2id[word]
            probs[i] += cnt / total

        return probs
    
    @lru_cache(maxsize=10_000)
    def get_prob_word(self, context: tuple[str], word: str):
        voc_len = len(self.vocabulary)
        counter = self.counter_seq.get(context, Counter())
        total = counter.total() + self.delta * voc_len

        p = counter[word] / total

        return p
    
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
        # print(norm_probs)
        norm_probs = norm_probs / norm_probs.sum()

        # семплирование
        sampled_i = np.random.choice(idx, p=norm_probs)
        return self.vocabulary[sampled_i]

    def check_prefix(self, prefix: list[str]) -> list[str]:
        # unk_i = prefix.count(vars.UNK) - 1
        unk_i = -1
        # print(tuple(prefix))
        # print(self.counter_seq.get(tuple(prefix)))
        while tuple(prefix) not in self.counter_seq and unk_i + 1 < len(prefix):
            prefix[unk_i+1] = vars.UNK
            unk_i += 1

        return prefix

    def predict_next(self, text: list[str], sample: str="top_p", p: float=0.8) -> tuple[str, np.array]:
        prefix = self.get_prefix(text)
        # prefix = self.check_prefix(prefix)

        # unk_i = prefix.count(vars.UNK) - 1
        # starts_with_unk = prefix[0] == vars.UNK

        # if not starts_with_unk:
        #     probs = self.get_probs(tuple(prefix))
        # else:
        #     unk_i = 0
        #     probs = np.zeros(len(self.vocabulary))
        #     probs += self.get_probs(tuple(prefix)) * self.alfas[len(prefix)-unk_i-1]
        #     while prefix.count(vars.UNK) < len(prefix):
        #         prefix[unk_i+1] = vars.UNK
        #         unk_i += 1
        #         probs += self.get_probs(tuple(prefix)) * self.alfas[len(prefix)-unk_i-1]

        probs = np.zeros(len(self.vocabulary))
        probs += self.get_probs(tuple(prefix)) * self.alfas[0]
        for i in range(prefix.count(vars.UNK)+1, self.n_gram-1):
            prefix[i] = vars.UNK
            probs += self.get_probs(tuple(prefix)) * self.alfas[i]

        probs = self.apply_temperature(probs)

        if sample == "top_p":
            pred_token = self.sample_top_p(probs, p)
        else:
            next_i = probs.argmax()
            pred_token = self.vocabulary[next_i]

        return pred_token, probs
    
    def perplexity(self, text: list[str]) -> float:
        pad = [vars.UNK] * (self.n_gram - 1)
        text = pad + text

        log_prob_sum = 0.0
        count = 0

        for i in range(self.n_gram - 1, len(text)):
            word = text[i]

            if not self.vocab2id.get(word):
                continue  # или считать как UNK

            prefix = text[i-self.n_gram+1:i]
            # print(prefix)
            # prefix = self.check_prefix(prefix)
            # print(prefix)

            # unk_i = prefix.count(vars.UNK) - 1

            # starts_with_unk = prefix[0] == vars.UNK

            # if not starts_with_unk:
            #     p = self.get_prob_word(tuple(prefix), word)
            # else:
            #     unk_i = 0
            #     p = 0
            #     p += self.get_prob_word(tuple(prefix), word) * self.alfas[len(prefix)-unk_i-1]
            #     # print(f"prefix: {prefix}, word: {word}, p: {p},")
            #     # time.sleep(1)
            #     while unk_i + 1 < len(prefix):
            #         prefix[unk_i+1] = vars.UNK
            #         unk_i += 1
            #         p += self.get_prob_word(tuple(prefix), word) * self.alfas[len(prefix)-unk_i-1]

            p = 0
            p += self.get_prob_word(tuple(prefix), word) * self.alfas[0]
            for i in range(prefix.count(vars.UNK)+1, self.n_gram-1):
                prefix[i] = vars.UNK
                p += self.get_prob_word(tuple(prefix), word) * self.alfas[i]

            # probs = np.array([p, 1-p])
            # probs = self.apply_temperature(probs)
            # p = probs[0]

            log_prob_sum += np.log(max(p, 1e-12))
            count += 1

            # print(f"log_prob_sum: {log_prob_sum} | count: {count}")

        # return np.exp(-log_prob_sum / count)
        return -log_prob_sum / count


def cross_validate(
        model: NGram, 
        val_text: list[str], 
        alfas: list[list[float]]
    ) -> tuple[NGram, list[float]]:

    train_start = time.time()

    model.train()

    print(f"train_time {time.time() - train_start}")

    print(f"Длина словаря: {len(model.vocabulary)}")
    print(f"Кол-во ngram: {len(model.counter_seq)}")
    # print(list(model.counter_seq.items())[:10])

    best_alfas = alfas[0]
    best_perplexity = None

    data = {}

    for alf in alfas:
        model.alfas = alf
        perplexity = model.perplexity(val_text)

        data[tuple(alf)] = perplexity

        print(f"alfas: {alf}")
        print(f"Perplexity: {perplexity}")

        if best_perplexity is None or perplexity < best_perplexity:
            best_perplexity = perplexity
            best_alfas = alf

    model.alfas = best_alfas

    return model, best_alfas, data

def cross_validate_vocab(
        train_text: list[str],
        val_text: list[str],
        n_gram: int,
        delta: float,
        alfas: list[float],
        min_c_list: list[int],
        min_C_list: list[int],
        T: float = 1.0,
    ):
    """
    Подбор min_c и min_C по perplexity на validation.
    """

    best_score = None
    best_params = None
    results = {}

    for min_c in min_c_list:
        for min_C in min_C_list:
            model = NGram(
                train_text=train_text,
                n_gram=n_gram,
                delta=delta,
                alfas=alfas,
                min_c=min_c,
                min_C=min_C,
                T=T,          # T здесь не влияет на perplexity (должно быть 1.0)
            )

            model.train()

            ppl = model.perplexity(val_text)
            results[(min_c, min_C)] = ppl

            print(
                f"min_c={min_c:2d}, min_C={min_C:2d} "
                f"-> perplexity={ppl:.4f}, "
                f"|V|={len(model.vocabulary)}, "
                f"|contexts|={len(model.counter_seq)}"
            )

            if best_score is None or ppl < best_score:
                best_score = ppl
                best_params = (min_c, min_C)

    return best_params, best_score, results




def clear_console():
    print("\033[2J\033[H", end="")

if __name__ == "__main__":
    os.environ['TCL_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
    os.environ['TK_LIBRARY'] = r'C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

    load_start = time.time()

    # with open(os.path.join(vars.DATASETS_ROOT, "quora.txt"), "r", encoding="UTF-8") as f:
    #     text = f.readlines()
    #     text = list(map(lambda x: x.strip() + " " + vars.EOS, text))

    #     split_border = int(0.9 * len(text))
    #     train, val = text[:split_border], text[split_border:]

    #     train_text = text_tokenize(" ".join(train))
    #     val_text = text_tokenize(" ".join(val))

    #     print(text[:50])

    data = pd.read_json(os.path.join(vars.DATASETS_ROOT, "arxivData.json"))
    lines = data.apply(lambda row: row['title'] + ' ; ' + row['summary'].replace("\n", ' ') + " " + vars.EOS, axis=1).tolist()

    print(len(lines))

    split_border = int(0.9 * len(lines))
    train, val = lines[:split_border], lines[split_border:]

    train_text = text_tokenize(" ".join(train))
    val_text = text_tokenize(" ".join(val))


    print(f"load_time {time.time() - load_start}")

    min_c = 3
    min_C = 3
    n_gram = 3
    delta = 0.1
    T = 0.5

    # alfas = np.random.random((5, n_gram))
    # print(alfas)
    # print(alfas.sum(axis=1))
    # alfas = alfas / alfas.sum(axis=1, keepdims=True)

    alfas = [
        [0.05, 0.15, 0.8],
        [0.1, 0.3, 0.6],
        [0.33, 0.34, 0.33],
        [0.6, 0.3, 0.1],
        [0.8, 0.15, 0.05],
    ]

    print(alfas)

    model = NGram(
        train_text=train_text,
        n_gram=n_gram,
        delta=delta,
        alfas=alfas[0],
        min_c=min_c,
        min_C=min_C,
        T=T
    )

    # print(f"Длина словаря: {len(model.vocabulary)}")
    # print(f"Кол-во ngram: {len(model.counter_seq)}")
        
    trained_model, best_alfas, data_map = cross_validate(
        model=model,
        val_text=val_text,
        alfas=alfas,
    )

    ax = sns.barplot(
        x=list(map(str, data_map.keys())),
        y=list(data_map.values()),
    )

    values = list(data_map.values())
    min_idx = np.argmin(values)

    for i, bar in enumerate(ax.patches):
        if i == min_idx:
            bar.set_color("green")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f")

    ax.set_title("Perplexity by different alfas")
    ax.set_xlabel("Alfas")
    ax.set_ylabel("Perplexity")

    plt.show()

    best_params, best_ppl, grid = cross_validate_vocab(
        train_text=train_text,
        val_text=val_text,
        n_gram=3,
        delta=0.1,
        alfas=best_alfas,
        min_c_list=[1, 3, 5],
        min_C_list=[1, 3, 5],
        T=1.0,   # важно: T=1 для perplexity
    )

    print(f"\nBEST: min_c={best_params[0]}, min_C={best_params[1]}")
    print(f"Perplexity={best_ppl:.4f}")

    df = pd.DataFrame(
        [
            {"min_c": k[0], "min_C": k[1], "perplexity": v}
            for k, v in grid.items()
        ]
    )

    pivot = df.pivot(index="min_c", columns="min_C", values="perplexity")

    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Perplexity vs vocabulary cleanup")
    plt.show()


    # train_start = time.time()

    # model.train()
    # trained_model = model

    # print(f"train_time {time.time() - train_start}")

    # print(trained_model.counter_seq.get((vars.UNK, "?")))
    # print(trained_model.counter_seq.get((vars.UNK, "?"), Counter()).get(vars.EOS))

    check_text = "The"
    check_text_tokenized = text_tokenize(check_text)

    clear_console()
    print(" ".join(check_text_tokenized))

    generate_start = time.time()

    while len(check_text_tokenized) < 100:
        pred_token, probs = trained_model.predict_next(check_text_tokenized)

        pairs = list(zip(trained_model.vocabulary, probs))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

        check_text_tokenized.append(pred_token)

        # os.system('cls' if os.name == 'nt' else 'clear')
        clear_console()
        print(" ".join(check_text_tokenized))
        # print(pairs_sorted[:5])
        # print(pairs_sorted[-5:])

    print(f"generate_time {time.time() - generate_start}")

    




    