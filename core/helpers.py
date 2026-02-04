import json
import os
import re
import time
import pandas as pd
from typing import Dict, List, Tuple, Iterable, Optional
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup

from ssau_api import SSAU_FILE
import vars
from core.models import NGram

def clear_console() -> None:
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")

def clean_text(text: str) -> str:
    if not text:
        return ""

    # 1. unicode escapes
    # try:
    #     text = bytes(text, "utf-8").decode("unicode_escape")
    # except Exception:
    #     pass

    # 2. HTML
    text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)

    # 3. лишние пробелы
    text = " ".join(text.split())

    return text

# =========================
# Tokenization
# =========================

def text_tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase word tokens.

    Args:
        text: Raw input text.

    Returns:
        List of tokens.
    """
    tokenizer = WordPunctTokenizer()
    return tokenizer.tokenize(text.lower())

def load_quora_dataset(
    dataset_root: str,
    split_ratio: float = 0.9,
) -> Tuple[List[str], List[str]]:
    """
    Load and split Quora dataset.

    Args:
        dataset_root: Path to datasets directory.
        split_ratio: Train/validation split ratio.

    Returns:
        (train_tokens, val_tokens)
    """
    file_path = os.path.join(dataset_root, "quora.txt")

    with open(file_path, "r", encoding="UTF-8") as f:
        lines = [
            line.strip() + " " + vars.EOS
            for line in f.readlines()
        ]

    split_border = int(split_ratio * len(lines))
    train_lines, val_lines = lines[:split_border], lines[split_border:]

    train_text = text_tokenize(" ".join(train_lines))
    val_text = text_tokenize(" ".join(val_lines))

    return train_text, val_text


def load_arxiv_dataset(
    dataset_root: str,
    split_ratio: float = 0.9,
) -> Tuple[List[str], List[str]]:
    """
    Load and split arXiv dataset (title + summary).

    Args:
        dataset_root: Path to datasets directory.
        split_ratio: Train/validation split ratio.

    Returns:
        (train_tokens, val_tokens)
    """
    file_path = os.path.join(dataset_root, "arxivData.json")

    data = pd.read_json(file_path)

    lines = (
        data.apply(
            lambda row: (
                f"{row['title']} ; "
                f"{row['summary'].replace(chr(10), ' ')} "
                f"{vars.EOS}"
            ),
            axis=1,
        )
        .tolist()
    )

    split_border = int(split_ratio * len(lines))
    train_lines, val_lines = lines[:split_border], lines[split_border:]

    train_text = text_tokenize(" ".join(train_lines))
    val_text = text_tokenize(" ".join(val_lines))

    return train_text, val_text

def load_dungeon_dataset(
    dataset_root: str = None,
    split_ratio: float = 0.9,
) -> str:
    """
    Load and split arXiv dataset (title + summary).

    Args:
        dataset_root: Path to datasets directory.
        split_ratio: Train/validation split ratio.

    Returns:
        (train_tokens, val_tokens)
    """
    with open(os.path.join(vars.DATASETS_ROOT, "dungeon_messages.txt"), encoding="utf-8") as f:
        lines = f.readlines()

    text = telegram_logs_to_text(lines)

    print(text[:500])

    return text

def load_ssau_dataset() -> Tuple[List[str], List[str]]:
    """
    Load and split arXiv dataset (title + summary).

    Args:
        dataset_root: Path to datasets directory.
        split_ratio: Train/validation split ratio.

    Returns:
        (train_tokens, val_tokens)
    """
    with open(SSAU_FILE, "r", encoding="UTF-8") as f:
        news = json.load(f)

    result = []

    for data in news:
        title = clean_text(data.get("title", ""))
        descr = clean_text(data.get("descr", ""))
        text = clean_text(data.get("pubText", ""))

        # print(title[:20])
        # print(descr[:20])
        # print(text[:20])

        # result.append(title + vars.PAD + descr + vars.PAD + text + vars.EOS)
        result.append(f"{title} {vars.PAD} {descr} {vars.PAD} {text}")

    return f" {vars.EOS} ".join(result)




def telegram_logs_to_text(lines: list[str]) -> str:
    line_re = re.compile(
        r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \d+: (.*)$"
    )
    messages = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = line_re.match(line)
        if not m:
            continue  # пропускаем битые строки

        msg = m.group(1).strip().lower()
        if msg:
            messages.append(msg)

    with open(os.path.join(vars.DATASETS_ROOT, "dungeon_messages_cleared.txt"), "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(msg + "\n")

    return f" {vars.EOS} ".join(messages)


def cross_validate(
        model: NGram, 
        val_text: list[str], 
        alfas: list[list[float]]
    ) -> tuple[NGram, list[float]]:

    train_start = time.time()

    model.train()
    model.filter_counter_seq()

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
        model = NGram(
            train_text=train_text,
            n_gram=n_gram,
            delta=delta,
            alfas=alfas,
            min_c=min_c,
            min_C=None,
            T=T,
        )

        model.train()

        base_counter_seq = model.counter_seq.copy()

        for min_C in min_C_list:
            model.min_C = min_C
            model.counter_seq = base_counter_seq
            model.filter_counter_seq()

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

def generate_example(model: NGram, tokenized_text: List[str], symbols_limit: int=100, stream: bool=True):
    generate_start = time.time()

    if stream:
        clear_console()
        print(" ".join(tokenized_text))

        while len(tokenized_text) < symbols_limit:
            pred_token, probs = model.predict_next(tokenized_text)

            pairs = list(zip(model.vocabulary, probs))
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

            tokenized_text.append(pred_token)

            # os.system('cls' if os.name == 'nt' else 'clear')
            clear_console()
            print(" ".join(tokenized_text))
            # print(pairs_sorted[:5])
            # print(pairs_sorted[-5:])
    else:
        while len(tokenized_text) < symbols_limit:
            pred_token, probs = model.predict_next(tokenized_text)

            pairs = list(zip(model.vocabulary, probs))
            pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

            tokenized_text.append(pred_token)

        print(" ".join(tokenized_text))

    print(f"generate_time {time.time() - generate_start}")

import random
import string

def char_noise(text, p=0.02):
    chars = list(text)
    i = 0
    while i < len(chars):
        if random.random() < p:
            op = random.choice(["delete", "swap", "replace"])

            if op == "delete":
                chars.pop(i)
                continue

            elif op == "swap" and i + 1 < len(chars):
                chars[i], chars[i+1] = chars[i+1], chars[i]

            elif op == "replace":
                chars[i] = random.choice(string.ascii_letters + " ")

        i += 1

    return "".join(chars)
