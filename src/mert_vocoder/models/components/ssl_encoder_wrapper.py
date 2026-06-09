from typing import Any, Protocol

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature


class FeatureExtractorLike(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> BatchFeature: ...


class SslEncoderWrapper(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        preprocessor: FeatureExtractorLike,
        sample_rate: int,
    ):
        super().__init__()

        self.model = model
        self.preprocessor = preprocessor
        self.sample_rate = sample_rate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        preprocessed_inputs = self.preprocessor(inputs, sampling_rate=self.sample_rate, return_tensors="pt")
        preprocessed_inputs = preprocessed_inputs.to(self.model.device)
        return self.model(**preprocessed_inputs, output_hidden_states=True).last_hidden_state
