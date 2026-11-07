from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str):
        chars = sorted(set(text))                          # unique chars, sorted
        stoi = {ch: i for i, ch in enumerate(chars)}       # char -> int
        itos = {i: ch for ch, i in stoi.items()}           # int -> char
        return stoi, itos

    def encode(self, text: str, stoi) -> list[int]:
        return [stoi[ch] for ch in text]

    def decode(self, ids, itos) -> str:
        return ''.join(itos[i] for i in ids)

