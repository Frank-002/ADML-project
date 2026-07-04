from abc import abstractmethod, ABC

import torch
from torch import nn


class Backbone(nn.Module, ABC):
    def __init__(
            self,
            model,
            device: torch.device,
            patch_size: int,
            trainable: bool = False,
    ):
        super().__init__()
        self.device = device
        self.model = model.to(self.device).train(trainable)
        self.patch_size = patch_size

    @abstractmethod
    def forward(
            self,
            image: torch.Tensor,
    ) -> torch.Tensor:
        pass
