from typing import List, Dict
    
class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        return [self._greedy_tokenize(str(n), vocab) for n in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        return len(self._greedy_tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        tokens = self._greedy_tokenize(text, vocab)
        words = text.split()
        if not words:
            return 0.0
        return round(len(tokens) / len(words), 4)

    def _greedy_tokenize(self, text, vocab):
        tokens = []
        i = 0
        while i < len(text):
            best = None
            # try longest substrings first
            for length in range(len(text) - i, 0, -1):
                substr = text[i:i + length]
                if substr in vocab:
                    best = substr
                    break
            if best is None:
                tokens.append(text[i])   # fallback: single char
                i += 1
            else:
                tokens.append(best)
                i += len(best)
        return tokens

