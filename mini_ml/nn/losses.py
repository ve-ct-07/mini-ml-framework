import numpy as np

from .modules.base import Module
from .autograd import Value


class BinaryCrossEntropyLoss(Module):

    def __init__(self, reduction='mean'):

        super().__init__()

        if reduction not in (
            'mean',
            'sum',
            'none'
        ):
            raise ValueError(
                "reduction must be "
                "'mean', 'sum', or 'none'"
            )

        self.reduction = reduction

    def __call__(self, logits, targets):

        targets = (
            targets.astype(np.float64)
            .reshape(-1, 1)
        )
        
        targets_np = np.asarray(
            targets,
            dtype=np.float64
        ).reshape(-1,1)

        probs = (
            logits
            .sigmoid()
            .clip(1e-12, 1 - 1e-12)
        )

        targets_val = Value(targets_np)

        loss = -(
            targets_val * probs.log()
            +
            (1 - targets_val)
            * (1 - probs).log()
        )

        if self.reduction == 'mean':
            return loss.mean()

        if self.reduction == 'sum':
            return loss.sum()

        return loss

    def __repr__(self):
        return (
            f"BinaryCrossEntropyLoss("
            f"reduction='{self.reduction}')"
        )


class CrossEntropyLoss(Module):

    def __init__(self, reduction='mean'):

        super().__init__()

        if reduction not in (
            'mean',
            'sum',
            'none'
        ):
            raise ValueError(
                "reduction must be "
                "'mean', 'sum', or 'none'"
            )

        self.reduction = reduction

    def __call__(self, logits, targets):

        batch_size, n_classes = logits.data.shape

        if targets.ndim != 1:
            raise ValueError(
                "target must be 1D class indices"
            )
        
        if len(targets) != logits.data.shape[0]:
            raise ValueError(
                "target size mismatch"
            )

        stable = logits - Value(
            np.max(
                logits.data,
                axis=1,
                keepdims=True
            )
        )

        log_probs = (
            stable
            -
            stable.exp()
            .sum(axis=1, keepdims=True)
            .log()
        )

        if np.any(targets < 0) or np.any(targets >= n_classes):
            raise ValueError(
                "Target contains invalid class indices."
            )

        one_hot = np.zeros(
            (batch_size, n_classes)
        )

        one_hot[
            np.arange(batch_size),
            targets
        ] = 1.0

        one_hot = Value(one_hot)

        loss = -(
            one_hot * log_probs
        ).sum(axis=1)

        if self.reduction == 'mean':
            return loss.mean()

        if self.reduction == 'sum':
            return loss.sum()

        return loss

    def __repr__(self):
        return (
            f"CrossEntropyLoss("
            f"reduction='{self.reduction}')"
        )
