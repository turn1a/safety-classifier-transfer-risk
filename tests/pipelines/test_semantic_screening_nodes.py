from types import SimpleNamespace

import numpy as np
import pytest
import torch

from transfer_risk.pipelines.semantic_screening.nodes import masked_mean_embeddings


def test_semantic_encoder_pooling_ignores_padding() -> None:
    class Encoder:
        def __call__(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                last_hidden_state=torch.tensor([[[2.0, 0.0], [0.0, 2.0], [100.0, 100.0]]])
            )

    vectors = masked_mean_embeddings(Encoder(), {"attention_mask": torch.tensor([[1, 1, 0]])})
    np.testing.assert_allclose(vectors, [[1, 1]])


def test_semantic_encoder_pooling_rejects_empty_windows() -> None:
    class Encoder:
        def __call__(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(last_hidden_state=torch.tensor([[[2.0, 0.0]]]))

    with pytest.raises(ValueError, match="empty"):
        masked_mean_embeddings(Encoder(), {"attention_mask": torch.tensor([[0]])})
