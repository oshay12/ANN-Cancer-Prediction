from math import prod
from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn import Conv2d, Flatten, Linear, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

NUM_CLASSES = 10
INPUT_SHAPE = (1, 28, 28)
OUT_CHANNELS = 1
BATCH_SIZE = 8
EPOCHS = 1


"""
# PyTorch Lightning vs. Keras

While PyTorch Lightning takes more code to setup, it makes clearer what is really "going on" by
explicitly laying out the training and validation steps, and where the loss, metrics, and logging
take place. Being dynamic and Pythonic (e.g. using object-oriented classes, with less overall
compilation to C-code), it is also easier to implement custom code, debug, and understand. So it is
increasingly popular among academic and other researchers.

One tradeoff, at least for simple models on a single GPU/CPU, is that PyTorch can be a bit slower.
However, when it comes to multiple CPUs/GPUs, performance has less to with the library (PyTorch,
Tensorflow) and more to do with careful optimizations specific to the problem at hand. Thus, while
there is no "best" library for deployment, PyTorch is hard to beat for experimentation and learning.



# On `Flatten` Layers

Convolution layers maintain the dimensionality of their inputs. That is, 2D inputs of shape
(in_channels, h, w) get mapped to 2D outputs of shape (out_channels, h', w'), even if h' and w' are
1. But in classification or regression, *at some point* we want to map a 2D input to a 1D output
(the predicted label or regression value).

In deep learning, there are two popular ways to perform this reduction: either flatten (unroll) the
2D input to a 1D vector, and use a Linear/Dense layer, or use global average pooling (GAP)
(https://paperswithcode.com/method/global-average-pooling). Flattening a d-dimensional image of size
(n_channels, n1, n2, ..., n_d) results in a vector of size (n_channels * n1 * n2 * ... * n_d). With
GAP, there is a drastic reduction, and depending on whether GAP maintains channels or not, the size
will be either (n_channels, 1) or (1,). There are various tradeoffs (both in terms of compute time
and model interpretation and capacity) for each. We use the classic flattening approach below.
"""


class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        # For odd-sized kernels with stride 1, "same" padding (keeping the output the same size as
        # the input) is achieved by making padding = kernel_size. For more details, see
        # https://arxiv.org/pdf/1603.07285.pdf#subsection.2.2.1. In Keras, you can just write
        # `padding='same'` and it will figure out what padding is needed. Because PyTorch doesn't
        # pre-compile, you have to tell PyTorch what padding is needed to maintain image size.
        k = 3
        p = k // 2  # note x // k is the same as int(x / k) or math.floor(x / k)
        self.model = Sequential(
            Conv2d(in_channels=1, out_channels=OUT_CHANNELS, kernel_size=k, padding=p, bias=True),
            ReLU(),  # Without a non-linear activation, a Conv2D layer is just linear
            Flatten(),  # see notes above
            Linear(in_features=OUT_CHANNELS * prod(INPUT_SHAPE), out_features=NUM_CLASSES),
        )
        # With 16 output channels, and since we padded to keep conv layer outputs the same size, we
        # have OUT_CHANNELS * INPUT_SIZE pixels (features) going into the final Linear layer.

    def forward(self, x: Tensor) -> Tensor:
        """The `forward` function defines how data flows through our model. Since we have a simple
        simple sequential model where data flows through layer after layer, we just pass the
        data through that sequential model here and let it handle the logic."""
        return self.model(x)

    def configure_optimizers(self) -> Optimizer:
        """The optimizer (Adam, RMSProp, SGD) has to update the model weights, or parameters. So
        we pass in those parameters to the optimizer here."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self._shared_step(batch, "train")[0]
        return loss  # we return the loss here for the optimizer to use to update weights

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        # no return is needed here because validation information should not be used in training!
        loss, acc = self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tuple[Tensor, Tensor]:
        return self._shared_step(batch, "test")

    def _shared_step(self, batch: Tuple[Tensor, Tensor], step: str) -> Tuple[Tensor, Tensor]:
        """Note the basic components of training, validation, and testing, are the same. We compute
        losses and metrics, log them, and return whaever values are needed."""
        x, target = batch
        prediction = self.model(x)
        acc = accuracy(prediction, target, task='binary')
        loss = cross_entropy(prediction, target)
        self.log(f"{step}_loss", loss)
        self.log(f"{step}_acc", acc)
        return loss, acc


if __name__ == "__main__":
    # PyTorch MNIST data download is already normalized to be in [0, 1]
    train_dataset = MNIST("", train=True, download=True, transform=transforms.ToTensor())
    test = MNIST("", train=False, download=True, transform=transforms.ToTensor())

    train, val = random_split(train_dataset, [50000, 10000], generator=None)

    # You can read about DataLoader and Trainer options in the documentation
    # https://pytorch.org/docs/stable/data.html
    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer
    train_loader = DataLoader(train, batch_size=32, num_workers=2)
    val_loader = DataLoader(val, batch_size=32, num_workers=2)
    test_loader = DataLoader(test, batch_size=32, num_workers=2)

    model = LightningMNISTClassifier()
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
