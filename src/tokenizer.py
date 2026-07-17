import json
import re


class MedicalTokenizer:

    def __init__(self):

        self.word2idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3,
        }

        self.idx2word = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<SOS>",
            3: "<EOS>",
        }

    def tokenize(self, text):

        text = text.lower()

        text = re.sub(r"[^a-z0-9 ]", "", text)

        return text.split()

    def build_vocab(self, texts, max_vocab_size=30000):
        from collections import Counter
        word_counts = Counter()
        chunk_size = 10000
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            chunk_text = " ".join(chunk).lower()
            # Apply regex and split in one go for the entire chunk
            chunk_text = re.sub(r"[^a-z0-9 ]", "", chunk_text)
            word_counts.update(chunk_text.split())

        # Select the most common words
        most_common = word_counts.most_common(max_vocab_size)

        for word, count in most_common:
            if word not in self.word2idx:
                index = len(self.word2idx)
                self.word2idx[word] = index
                self.idx2word[index] = word

    def encode(self, text, max_length=64):

        words = self.tokenize(text)

        ids = [self.word2idx["<SOS>"]]

        for word in words:

            ids.append(
                self.word2idx.get(
                    word,
                    self.word2idx["<UNK>"]
                )
            )

        ids.append(self.word2idx["<EOS>"])

        if len(ids) < max_length:

            ids.extend(
                [self.word2idx["<PAD>"]] *
                (max_length - len(ids))
            )

        else:

            ids = ids[:max_length]

        return ids

    def decode(self, ids):

        words = []

        for idx in ids:

            word = self.idx2word.get(idx, "<UNK>")

            if word in ["<PAD>", "<SOS>", "<EOS>"]:
                continue

            words.append(word)

        return " ".join(words)

    def vocab_size(self):

        return len(self.word2idx)

    def save_vocab(self, path):

        with open(path, "w", encoding="utf-8") as f:

            json.dump(self.word2idx, f, indent=4)

    def load_vocab(self, path):

        with open(path, "r", encoding="utf-8") as f:

            self.word2idx = json.load(f)

        self.idx2word = {
            int(v): k
            for k, v in self.word2idx.items()
        }