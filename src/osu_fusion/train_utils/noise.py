import random

import torch


def pyramid_noise(noise: torch.Tensor, device: torch.device, iterations: int = 6, discount: int = 0.4) -> torch.Tensor:
    b, d, n = noise.shape
    u = torch.nn.Upsample(size=n, mode="linear", align_corners=False)
    for i in range(iterations):
        r = random.random() * 2 + 2
        n = max(1, int(n / (r**i)))
        noise += u(torch.randn((b, d, n), device=device)) * (discount**i)
        if n == 1:
            break  # no need to continue if we are down to 1
    return noise / noise.std()  # Scale to unit variance
