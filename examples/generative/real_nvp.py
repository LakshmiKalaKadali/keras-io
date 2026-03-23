"""
Title: Density estimation using Real NVP
Authors: [Mandolini Giorgio Maria](https://www.linkedin.com/in/giorgio-maria-mandolini-a2a1b71b4/), [Sanna Daniele](https://www.linkedin.com/in/daniele-sanna-338629bb/), [Zannini Quirini Giorgio](https://www.linkedin.com/in/giorgio-zannini-quirini-16ab181a0/)
Date created: 2020/08/10
Last modified: 2026/03/23
Description: Estimating the density distribution of the "double moon" dataset.
Accelerator: GPU
Converted to Keras 3 by: [LakshmiKalaKadali](https://github.com/LakshmiKalaKadali)
"""

"""
## Introduction

The aim of this work is to map a simple distribution - which is easy to sample
and whose density is simple to estimate - to a more complex one learned from the data.
This kind of generative model is also known as "normalizing flow".

In order to do this, the model is trained via the maximum
likelihood principle, using the "change of variable" formula.

We will use an affine coupling function. We create it such that its inverse, as well as
the determinant of the Jacobian, are easy to obtain (more details in the referenced paper).

**Reference:**

[Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)
"""

"""
## Setup

"""
import os

# Set backend to JAX, PyTorch, or TensorFlow
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers, ops, regularizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

"""
## Load the data
"""

data = make_moons(3000, noise=0.05)[0].astype("float32")
norm = layers.Normalization()
norm.adapt(data)
normalized_data = norm(data)

"""
## Affine coupling layer
"""

# Creating a custom layer with keras API.
output_dim = 256
reg = 0.01


def Coupling(input_shape):
    input_layer = layers.Input(shape=(input_shape,))

    def mlp(x):
        for _ in range(4):
            x = layers.Dense(
                256, activation="relu", kernel_regularizer=regularizers.l2(0.01)
            )(x)
        return x

    # Scale (s) and Translation (t)
    shared = mlp(input_layer)
    s = layers.Dense(input_shape, activation="tanh")(shared)
    t = layers.Dense(input_shape, activation="linear")(shared)
    return keras.Model(inputs=input_layer, outputs=[s, t])


"""
## Real NVP
"""


class RealNVP(keras.Model):
    def __init__(self, num_coupling_layers, **kwargs):
        super().__init__(**kwargs)
        self.num_coupling_layers = num_coupling_layers
        self.masks = ops.convert_to_tensor(
            np.array([[0, 1], [1, 0]] * (num_coupling_layers // 2), dtype="float32")
        )
        self.coupling_layers = [Coupling(2) for _ in range(num_coupling_layers)]

    def log_prob_std_normal(self, z):
        d = ops.cast(ops.shape(z)[-1], "float32")
        log2pi = ops.cast(np.log(2.0 * np.pi), "float32")
        return -0.5 * (d * log2pi + ops.sum(ops.square(z), axis=-1))

    def call(self, x, training=False):
        log_det_inv = 0
        direction = -1.0 if training else 1.0
        layer_indices = range(self.num_coupling_layers)
        if training:
            layer_indices = reversed(layer_indices)

        for i in layer_indices:
            x_masked = x * self.masks[i]
            reversed_mask = 1.0 - self.masks[i]
            s, t = self.coupling_layers[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1.0) / 2.0
            x = (
                reversed_mask
                * (x * ops.exp(direction * s) + direction * t * ops.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * ops.sum(s, axis=1)
        return x, log_det_inv

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        z, logdet = y_pred
        log_likelihood = self.log_prob_std_normal(z) + logdet
        main_loss = -ops.mean(log_likelihood)

        # Manually sum the L2 losses from the coupling layers
        # Ensure reg_losses is a Keras tensor, even if self.losses is empty
        if self.losses:
            # Stack the losses into a single tensor and then sum them up
            reg_losses = ops.sum(ops.stack(self.losses))
        else:
            reg_losses = ops.convert_to_tensor(0.0, dtype="float32")

        return main_loss + reg_losses


"""
## Model training
"""

model = RealNVP(num_coupling_layers=6)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

indices = np.arange(len(normalized_data))
np.random.shuffle(indices)
normalized_data = normalized_data[indices]

# Now fit
history = model.fit(
    normalized_data, batch_size=256, epochs=300, verbose=2, validation_split=0.2
)

"""
## Performance evaluation
"""

plt.figure(figsize=(15, 10))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.legend(["train", "validation"], loc="upper right")
plt.ylabel("loss")
plt.xlabel("epoch")

# From data to latent space.
z, _ = model(normalized_data, training=True)  # Ensure training=True for data->latent
z = ops.convert_to_numpy(z)

# From latent space to data.
samples = keras.random.normal(shape=(3000, 2))  # Correctly sample from standard normal
x, _ = model(
    samples, training=False
)  # Use model's call method for generation (latent->data)
x = ops.convert_to_numpy(x)

f, axes = plt.subplots(2, 2)
f.set_size_inches(20, 15)

axes[0, 0].scatter(normalized_data[:, 0], normalized_data[:, 1], color="r")
axes[0, 0].set(title="Inference data space X", xlabel="x", ylabel="y")
axes[0, 1].scatter(z[:, 0], z[:, 1], color="r")
axes[0, 1].set(title="Inference latent space Z", xlabel="x", ylabel="y")
axes[0, 1].set_xlim([-3.5, 4])
axes[0, 1].set_ylim([-4, 4])
axes[1, 0].scatter(samples[:, 0], samples[:, 1], color="g")
axes[1, 0].set(title="Generated latent space Z", xlabel="x", ylabel="y")
axes[1, 1].scatter(x[:, 0], x[:, 1], color="g")
axes[1, 1].set(title="Generated data space X", xlabel="x", ylabel="y")
axes[1, 1].set_xlim([-2, 2])
axes[1, 1].set_ylim([-2, 2])

"""
## Relevant Chapters from Deep Learning with Python
- [Chapter 17: Image generation](https://deeplearningwithpython.io/chapters/chapter17_image-generation)
"""
