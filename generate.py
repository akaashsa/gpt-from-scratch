import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        result = []

        for _ in range(new_chars):
            # 1. Crop context so it fits the model's attention window
            if context.shape[1] > context_length:
                context = context[:, -context_length:]

            # 2. Forward pass -> logits of shape (1, T, vocab_size)
            logits = model(context)

            # 3. Only the last position predicts the NEXT token
            last_logits = logits[:, -1, :]           # (1, vocab_size)

            # 4. Convert to probabilities
            probs = nn.functional.softmax(last_logits, dim=-1)

            # 5. Sample one token from the distribution
            next_token = torch.multinomial(probs, 1)  # (1, 1)

            # 6. Append to context, decode char, keep going
            context = torch.cat((context, next_token), dim=-1)
            result.append(int_to_char[next_token.item()])

        return ''.join(result)