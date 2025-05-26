import logging
import math

import beartype
import numpy as np
import sklearn.base
import sklearn.utils.validation
import torch
import torch.nn
import torch.utils.data

from . import helpers


@beartype.beartype
class LinearProbeClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """
    LayerNorm + Linear soft-max head trained with AdamW on fixed feature vectors.
    """

    def __init__(
        self,
        n_steps: int = 10_000,
        batch_size: int = 2048,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        device: str = "cpu",
    ):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.logger = logging.getLogger("linear-probe-clf")

    # fit
    def fit(self, X, y):
        x, y = sklearn.utils.validation.check_X_y(X, y, dtype=np.float32, order="C")
        x_dim, n_cls = x.shape[1], int(np.max(y) + 1)

        # model
        self.model_ = torch.nn.Sequential(
            torch.nn.LayerNorm(x_dim, elementwise_affine=True),
            torch.nn.Linear(x_dim, n_cls, bias=True),
        ).to(self.device)
        self.model_.train()

        # param-wise wd: skip bias & LN γ,β
        decay, no_decay = [], []
        for n, p in self.model_.named_parameters():
            (decay if p.ndim > 1 else no_decay).append(p)
        opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )
        lr_scheduler = CosineWarmup(
            init=self.lr,
            max=self.lr,
            final=self.lr * 1e-2,
            n_warmup_steps=0,
            n_steps=self.n_steps,
        )

        # data
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x), torch.from_numpy(y).long()
        )
        loader = torch.utils.data.DataLoader(
            dataset, self.batch_size, shuffle=True, pin_memory=True, drop_last=False
        )

        scaler = torch.amp.GradScaler()

        # training loop
        it = helpers.infinite(loader)
        for step in helpers.progress(
            range(self.n_steps), every=self.n_steps // 100, desc="sgd"
        ):
            xb, yb = next(it)
            xb, yb = xb.to(self.device), yb.to(self.device)
            with torch.amp.autocast(self.device):
                logits = self.model_(xb)
                loss = torch.nn.functional.cross_entropy(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            lr = lr_scheduler.step()
            for param_group in opt.param_groups:
                param_group["lr"] = lr
            opt.zero_grad(set_to_none=True)

            if step % (self.n_steps // 100) == 0:
                self.logger.info(
                    "step: %d/%d (%.1f%%) loss: %.6f lr: %.3g",
                    step,
                    self.n_steps,
                    (step / self.n_steps * 100),
                    loss.item(),
                    lr,
                )

        self.model_.eval()
        return self

    # predict
    def predict(self, X):
        sklearn.utils.validation.check_is_fitted(self, ["model_"])
        x = sklearn.utils.validation.check_array(X, dtype=np.float32, order="C")
        xb = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            logits = self.model_(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds

    # predict_proba
    def predict_proba(self, X):
        sklearn.utils.validation.check_is_fitted(self, ["model_"])
        x = sklearn.utils.validation.check_array(X, dtype=np.float32, order="C")
        xb = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(self.model_(xb), dim=1)
        return probs.cpu().numpy()


@beartype.beartype
class Scheduler:
    def step(self) -> float:
        err_msg = f"{self.__class__.__name__} must implement step()."
        raise NotImplementedError(err_msg)


@beartype.beartype
class Linear(Scheduler):
    def __init__(self, init: float, final: float, n_steps: int):
        self.init = init
        self.final = final
        self.n_steps = n_steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        return self.init + (self.final - self.init) * (self._step / self.n_steps)


@beartype.beartype
class CosineWarmup(Scheduler):
    def __init__(
        self,
        *,
        init: float,
        max: float,
        final: float,
        n_warmup_steps: int,
        n_steps: int,
    ):
        self.init = init
        self.max = max
        self.final = final
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = n_steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        if self._step < self.n_warmup_steps:
            # Linear warmup
            return self.init + (self.max - self.init) * (
                self._step / self.n_warmup_steps
            )

        # Cosine decay.
        return self.final + 0.5 * (self.max - self.final) * (
            1
            + math.cos(
                (self._step - self.n_warmup_steps)
                / (self.n_steps - self.n_warmup_steps)
                * math.pi
            )
        )


def _plot_example_schedules():
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

    n_steps = 1000
    xs = np.arange(n_steps)

    schedule = Linear(0.1, 0.9, n_steps)
    ys = [schedule.step() for _ in xs]

    ax1.plot(xs, ys)
    ax1.set_title("Linear")

    schedule = CosineWarmup(0.1, 1.0, 0.0, 0, n_steps)
    ys = [schedule.step() for _ in xs]

    ax2.plot(xs, ys)
    ax2.set_title("CosineWarmup")

    fig.tight_layout()
    fig.savefig("schedules.png")


if __name__ == "__main__":
    _plot_example_schedules()
