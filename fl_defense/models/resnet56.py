"""CIFAR ResNet-56 v1 with projection shortcuts at stage transitions.

The main path has 1 + 3 * 9 * 2 convolutional layers and one classifier
(6n + 2 = 56, n = 9). Two 1x1 shortcut projections are not counted in this
depth convention. Inputs are channels-last 32x32 RGB images, preprocessed by
the caller, as for the CNN baseline.

Reference: https://arxiv.org/abs/1512.03385
"""

import keras
from keras import layers


def _conv_bn(x, filters, kernel_size, stride, name):
    """Apply a bias-free convolution followed by batch normalization."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        data_format="channels_last",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name=f"{name}_conv",
    )(x)
    return layers.BatchNormalization(axis=-1, name=f"{name}_bn")(x)


def _residual_block(x, filters, stride, name):
    """Build a post-activation basic block (Conv-BN-ReLU-Conv-BN-Add-ReLU)."""
    shortcut = x
    y = _conv_bn(x, filters, 3, stride, f"{name}_1")
    y = layers.ReLU(name=f"{name}_1_relu")(y)
    y = _conv_bn(y, filters, 3, 1, f"{name}_2")

    if stride != 1 or x.shape[-1] != filters:
        shortcut = _conv_bn(shortcut, filters, 1, stride, f"{name}_shortcut")

    y = layers.Add(name=f"{name}_add")([y, shortcut])
    return layers.ReLU(name=f"{name}_out")(y)


def load_model(learning_rate: float = 0.001):
    """Return a compiled ResNet-56 for CIFAR-10 with integer class labels.

    Matches the CNN baseline's factory signature, Adam optimizer, sparse
    categorical cross-entropy loss, and accuracy metric. All components are
    standard Keras layers, so .keras saves need no custom objects to reload.
    """
    inputs = keras.Input(shape=(32, 32, 3), name="image")
    x = _conv_bn(inputs, 16, 3, 1, "stem")
    x = layers.ReLU(name="stem_relu")(x)

    for stage, filters in enumerate((16, 32, 64), start=1):
        for block in range(1, 10):
            # Halve spatial dimensions only at the start of stages 2 and 3.
            stride = 2 if stage > 1 and block == 1 else 1
            x = _residual_block(x, filters, stride, f"stage{stage}_block{block}")

    x = layers.GlobalAveragePooling2D(
        data_format="channels_last", name="global_average_pool"
    )(x)
    outputs = layers.Dense(10, activation="softmax", name="classifier")(x)
    model = keras.Model(inputs, outputs, name="resnet56")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
