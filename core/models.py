import os
import time
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functools import lru_cache
from collections import Counter, defaultdict

import vars

# =========================
# N-gram Language Model
# =========================

class NGram:
    """
    Interpolated N-gram language model with vocabulary pruning,
    delta smoothing, and temperature-controlled sampling.
    """

    def __init__(
        self,
        train_text: List[str],
        n_gram: int = 3,
        delta: float = 0.1,
        alfas: np.ndarray = np.array([0.8, 0.15, 0.05]),
        min_c: int = 3,
        min_C: int = 3,
        T: float = 1.0,
    ) -> None:
        """
        Initialize N-gram model.

        Args:
            train_text: Tokenized training text.
            n_gram: N-gram order.
            delta: Additive smoothing parameter.
            alfas: Interpolation weights (must sum to 1).
            min_c: Minimum word frequency for vocabulary.
            min_C: Minimum context count for keeping n-grams.
            T: Sampling temperature (used only in generation).
        """
        self.n_gram = n_gram
        self.delta = delta
        self.alfas = alfas
        self.min_c = min_c
        self.min_C = min_C
        self.T = T

        self.counter_seq = defaultdict(Counter)

        self.build_vocab_and_replace_unk(train_text)

    # ---------------------
    # Vocabulary
    # ---------------------

    def build_vocab_and_replace_unk(self, text: List[str]) -> None:
        """
        Build vocabulary using min_c cutoff and replace OOV tokens with UNK.

        Args:
            text: Training tokens.
        """
        word_counts = Counter(text)

        vocab = {w for w, c in word_counts.items() if c >= self.min_c}
        vocab.add(vars.UNK)

        self.train_text = [
            w if w in vocab else vars.UNK for w in text
        ]

        self.vocabulary = list(vocab)
        self.vocab2id = {
            w: i for i, w in enumerate(self.vocabulary)
        }

    # ---------------------
    # Training
    # ---------------------

    def train(self) -> None:
        """
        Build full n-gram counters (including UNK-backed contexts).
        """
        pad = [vars.UNK] * (self.n_gram - 1)
        text = pad + self.train_text

        for i in range(self.n_gram - 1, len(text)):
            word = text[i]
            base_context = text[i - self.n_gram + 1:i]

            for j in range(self.n_gram):
                context = tuple([vars.UNK] * j + base_context[j:])
                self.counter_seq[context][word] += 1

    def filter_counter_seq(self) -> None:
        """
        Remove contexts with insufficient total counts (min_C).
        """
        if self.min_C is None:
            return

        self.counter_seq = {
            ctx: cnt
            for ctx, cnt in self.counter_seq.items()
            if cnt.total() >= self.min_C
        }

    # ---------------------
    # Probability
    # ---------------------

    @lru_cache(maxsize=100_000)
    def get_probs(self, context: Tuple[str, ...]) -> np.ndarray:
        """
        Return full conditional distribution P(w | context).

        Args:
            context: Context tuple.

        Returns:
            Probability vector over vocabulary.
        """
        V = len(self.vocabulary)
        counter = self.counter_seq.get(context, Counter())
        total = counter.total() + self.delta * V

        probs = np.full(V, self.delta / total)

        for word, cnt in counter.items():
            probs[self.vocab2id[word]] += cnt / total

        return probs

    @lru_cache(maxsize=10_000)
    def get_prob_word(self, context: Tuple[str, ...], word: str) -> float:
        """
        Return P(word | context).

        Args:
            context: Context tuple.
            word: Target word.

        Returns:
            Probability value.
        """
        V = len(self.vocabulary)
        counter = self.counter_seq.get(context, Counter())
        total = counter.total() + self.delta * V

        return counter[word] / total

    # ---------------------
    # Sampling
    # ---------------------

    def apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to a probability distribution.

        Args:
            probs: Probability vector.

        Returns:
            Temperature-adjusted probabilities.
        """
        if self.T <= 0:
            raise ValueError("Temperature must be > 0")

        if self.T == 1.0:
            return probs

        probs = np.maximum(probs, 1e-12)
        probs_T = probs ** (1.0 / self.T)
        return probs_T / probs_T.sum()

    def sample_top_p(self, probs: np.ndarray, p: float = 0.8) -> str:
        """
        Sample a token using nucleus (top-p) sampling.

        Args:
            probs: Probability vector.
            p: Cumulative probability threshold.

        Returns:
            Sampled token.
        """
        idx = np.argsort(probs)[::-1]
        cum_probs = np.cumsum(probs[idx])

        cut = np.searchsorted(cum_probs, p) + 1
        idx = idx[:cut]

        norm_probs = probs[idx]
        norm_probs /= norm_probs.sum()

        sampled_i = np.random.choice(idx, p=norm_probs)
        return self.vocabulary[sampled_i]

    # ---------------------
    # Inference
    # ---------------------

    def get_prefix(self, context: List[str]) -> List[str]:
        """
        Extract padded prefix for prediction.

        Args:
            context: Token history.

        Returns:
            Prefix of length n-1.
        """
        prefix = context[-(self.n_gram - 1):]
        return [vars.UNK] * (self.n_gram - 1 - len(prefix)) + prefix

    def predict_next(
        self,
        text: List[str],
        sample: str = "top_p",
        p: float = 0.8,
    ) -> Tuple[str, np.ndarray]:
        """
        Predict next token.

        Args:
            text: Context tokens.
            sample: Sampling strategy.
            p: Top-p threshold.

        Returns:
            (predicted token, probability distribution)
        """
        prefix = self.get_prefix(text)

        probs = np.zeros(len(self.vocabulary))
        probs += self.get_probs(tuple(prefix)) * self.alfas[0]

        for i in range(prefix.count(vars.UNK) + 1, self.n_gram - 1):
            prefix[i] = vars.UNK
            probs += self.get_probs(tuple(prefix)) * self.alfas[i]

        probs = self.apply_temperature(probs)

        if sample == "top_p":
            return self.sample_top_p(probs, p), probs

        return self.vocabulary[int(probs.argmax())], probs

    # ---------------------
    # Evaluation
    # ---------------------

    def perplexity(self, text: List[str]) -> float:
        """
        Compute average negative log-likelihood (perplexity proxy).

        Args:
            text: Validation tokens.

        Returns:
            Mean negative log-probability.
        """
        pad = [vars.UNK] * (self.n_gram - 1)
        text = pad + text

        log_prob_sum = 0.0
        count = 0

        for i in range(self.n_gram - 1, len(text)):
            word = text[i]

            if word not in self.vocab2id:
                continue

            prefix = text[i - self.n_gram + 1:i]

            p = self.get_prob_word(tuple(prefix), word) * self.alfas[0]

            for j in range(prefix.count(vars.UNK) + 1, self.n_gram - 1):
                prefix[j] = vars.UNK
                p += self.get_prob_word(tuple(prefix), word) * self.alfas[j]

            log_prob_sum += np.log(max(p, 1e-12))
            count += 1

        return -log_prob_sum / count
