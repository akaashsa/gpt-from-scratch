import torch
from typing import Tuple

class Solution:
    def create_batches(self, data, context_length, batch_size):
        torch.manual_seed(0)
        ix = torch.randint(len(data) - context_length, (batch_size,))
        x = torch.stack([data[i:i + context_length] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + context_length] for i in ix])
        return x, y