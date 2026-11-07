import torch
import torch.nn as nn
import torch.nn.functional as F

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int,
              context_length: int, batch_size: int, lr: float) -> float:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            # 1. Reproducible batch sampling per epoch
            torch.manual_seed(epoch)

            # 2. Random start indices in valid range
            ix = torch.randint(len(data) - context_length, (batch_size,))

            # 3. Build X and Y (Y shifted right by 1)
            x = torch.stack([data[i:i + context_length] for i in ix])
            y = torch.stack([data[i + 1:i + 1 + context_length] for i in ix])

            # 4. Forward pass
            logits = model(x)             # (B, T, C)
            B, T, C = logits.shape

            # 5-6. Reshape + cross entropy
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

            # 7. Backward + update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)