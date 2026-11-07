import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        torch.manual_seed(0)

        # 1. Split raw text into tokens (words)
        tokens = raw_dataset.split()

        # 2. Sample random start indices.
        #    Valid starts: 0 .. (len - context_length - 1)
        #    torch.randint high is exclusive, so pass len - context_length.
        starts = torch.randint(
            low=0,
            high=len(tokens) - context_length,
            size=(batch_size,),
        ).tolist()

        # 3. Build X (input) and Y (shifted-by-one labels)
        X, Y = [], []
        for i in starts:
            X.append(tokens[i : i + context_length])
            Y.append(tokens[i + 1 : i + 1 + context_length])

        return X, Y