"""Adapted from https://keras.io/examples/vision/mnist_convnet/, by Francois Chollet"""
import numpy as np
import keras

from keras import Input, Sequential
from keras.layers import Conv2D, Dense, Flatten, ReLU

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)
OUT_CHANNELS = 1
BATCH_SIZE = 8
EPOCHS = 1

"""
Note that in Tensorflow we don't specify as much shape information (in_channels, out_channels) as
in PyTorch. This is because Tensorflow models must be pre-compiled, so Tensorflow can look at all
the layers, and figure out the shapes needed for each layer and assign the layers those shapes (if
valid shapes are possible). The *cost* is that your model might work without you *really understanding
why*, and that error messages are much more cryptic.

You can avoid some of this by using Tensorflow "Eager execution"
(https://www.tensorflow.org/guide/eager). It is probably worth mentioning this was (probably) mostly
developed to compete with PyTorch, and that the examples you search for online will often assume you
are *not* in eager execution mode.



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

if __name__ == "__main__":
    model = Sequential(
        [
            # The Input layer lets Keras figure out all the shapes and sizes you need. It isn't
            # always necessary, but is almost always a good idea for basic models.
            Input(shape=INPUT_SHAPE),
            # Note we don't have to calculate the padding in Keras, due to precompilation.
            Conv2D(OUT_CHANNELS, kernel_size=3, padding="same", use_bias=True),
            ReLU(),  # Without a non-linear activation, a Conv2D layer is just linear
            Flatten(),  # see notes above
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.summary()  # print some model information. Done by default in PyTorch Lightning.

    # The "compile" step below is unique to Tensorflow / Keras. The model code above only
    # *blueprints* the actual model, and that blueprint is read and compiled to fast C / C++ code.
    # However, note that, like in PyTorch, we must configure the optimizer, and specify metrics and
    # loss functions.
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Keras by default does not scale MNIST, so we need to scale and convert to float. The default
    # MNIST data also doesn't include channel information, so we add it with np.expand_dims. Channel
    # dimensions of size 1, or batch sizes of 1 and etc. are constant pain points in deep learning.
    # You always have to add / subtract dimensions as needed for things to work.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # This is a Tensorflow / Keras oddity, in that classification targets need to be identified as a
    # "categorical" (one-hot) type to prevent certain errors resulting from the loss function
    # calculations. That is, when the loss is "categorical_crossentropy", the targets must be be
    # one-hot ("categorical") vectors, and when the loss is "sparse_categorical_crossentropy"
    # instead the targets can be integer label vectors (e.g. [1, 8, 7, 1]). Because PyTorch is
    # dynamic, it can (usually) figure it out, and you never worry about this.
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    model.fit(
        x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1
    )  # train
    score = model.evaluate(x_test, y_test, verbose=1)  # test
