"""GAN implementation."""

import math
import os
from datetime import datetime

import h5py
import numpy as np
import tensorflow.keras as keras
from PIL import Image

import ganji.datasets
import ganji.project

Sequential = keras.models.Sequential

Dense = keras.layers.Dense
Activation = keras.layers.Activation
Reshape = keras.layers.Reshape

BatchNormalization = keras.layers.BatchNormalization
UpSampling2D = keras.layers.UpSampling2D
Convolution2D = keras.layers.Convolution2D
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
AveragePooling2D = keras.layers.AveragePooling2D

LeakyReLU = keras.layers.LeakyReLU
Flatten = keras.layers.Flatten
Dropout = keras.layers.Dropout

Adam = keras.optimizers.Adam
SGD = keras.optimizers.SGD
RMSprop = keras.optimizers.RMSprop


def generator_model(n):
    model = Sequential()
    model.add(Dense(1024, input_dim=100))
    model.add(Activation("tanh"))
    model.add(Dense(128 * n * n))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("tanh"))
    model.add(Reshape((n, n, 128), input_shape=(128 * n * n,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(Activation("tanh"))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding="same"))
    model.add(Activation("tanh"))
    return model


def discriminator_model(n):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding="same", input_shape=(4 * n, 4 * n, 1)))
    model.add(Activation("tanh"))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation("tanh"))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("tanh"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0] : (i + 1) * shape[0], j * shape[1] : (j + 1) * shape[1]] = img[:, :, 0]
    return image


def load_image_data(dir, config):
    path = os.path.join(dir, "dataset.npy")
    if os.path.exists(path):
        return np.load(path)
    else:
        codepoints = ganji.datasets.find_codepoints(config.codepoint_set)
        font_index = 0 if config.font_index is None else config.font_index
        thickness_quantiles = (config.thickness_quantile_min, config.thickness_quantile_max)
        data = ganji.datasets.load_data_for_gan(
            codepoints, config.font, 4 * config.unit, font_index=font_index, thickness_quantiles=thickness_quantiles
        )
        np.save(path, data)
        return data


def train(dir):
    # prepare
    for subdirname in ["training", "models"]:
        subdir = os.path.join(dir, subdirname)
        if not os.path.isdir(subdir):
            os.mkdir(subdir)

    config, state = ganji.project.load_metadata(dir)

    batch_size = config.batch_size
    epoch_end = config.epoch_end
    n = config.unit
    epoch_start = state.epoch

    log_path = os.path.join(dir, "log.hdf5")
    if not os.path.exists(log_path):
        with h5py.File(log_path, "a") as f:
            f.create_dataset("epoch", (1,), dtype=np.int32)
            f.create_dataset("g_loss", (epoch_end + 1,), maxshape=(None,), dtype=np.float32)
            f.create_dataset("d_loss", (epoch_end + 1,), maxshape=(None,), dtype=np.float32)

    x_train = load_image_data(dir, config)
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    batch_count = int(x_train.shape[0] / batch_size)
    print("Number of batches", batch_count)

    d = discriminator_model(n)
    g = generator_model(n)
    d_on_g = generator_containing_discriminator(g, d)
    if os.path.exists(os.path.join(dir, "models", "generator.index")):
        g.load_weights(os.path.join(dir, "models", "generator"))
    d_on_g.compile(loss="binary_crossentropy", optimizer="SGD")
    d.trainable = True
    d.compile(loss="binary_crossentropy", optimizer="ADAM")
    if os.path.exists(os.path.join(dir, "models", "discriminator.index")):
        d.load_weights(os.path.join(dir, "models", "discriminator"))

    # train
    for epoch in range(epoch_start, epoch_end):
        print("Epoch is", epoch)

        state.epoch = epoch
        state.update_time = datetime.now().timestamp()

        g_loss = None
        d_loss = None
        for index in range(batch_count):
            # train discriminator
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            image_batch = x_train[index * batch_size : (index + 1) * batch_size]
            # image_batch = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
            generated_images = g.predict(noise, verbose=0)
            x = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size, dtype=np.bool)
            d_loss = d.train_on_batch(x, y)
            print(f"batch: {index:d}, d_loss: {d_loss:f}")
            # train generator
            noise = np.random.uniform(-1, 1, (batch_size, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, np.array([1] * batch_size, dtype=np.bool))
            d.trainable = True
            print(f"batch: {index:d}, g_loss: {g_loss:f}")
            if index == batch_count - 1:
                image = combine_images(generated_images)
                image = -image * 127.5 + 127.5
                image_path = os.path.join(dir, "training", f"{epoch:06d}.png")
                Image.fromarray(image.astype(np.uint8)).save(image_path)

        with h5py.File(log_path, "a") as log_file:
            log_file["epoch"][0] = epoch + 1
            log_file["g_loss"][epoch] = g_loss
            log_file["d_loss"][epoch] = d_loss

        if epoch % 100 == 0 or epoch == epoch_end - 1:
            g.save_weights(os.path.join(dir, "models", f"generator_{epoch:06d}"), True)
            d.save_weights(os.path.join(dir, "models", f"discriminator_{epoch:06d}"), True)

        if epoch % 10 == 0 or epoch == epoch_end - 1:
            g.save_weights(os.path.join(dir, "models", "generator"), True)
            d.save_weights(os.path.join(dir, "models", "discriminator"), True)
            state.epoch = epoch + 1
            ganji.project.dump_state(dir, state)


def generate(dir, *, epoch=None, nice=False):
    for subdirname in ["generated"]:
        subdir = os.path.join(dir, subdirname)
        if not os.path.isdir(subdir):
            os.mkdir(subdir)

    config = ganji.project._load_config(dir)
    batch_size = config.batch_size
    n = config.unit

    g = generator_model(n)
    g.compile(loss="binary_crossentropy", optimizer="SGD")
    if epoch is None:
        g.load_weights(os.path.join(dir, "models", "generator"))
    else:
        g.load_weights(os.path.join(dir, "models", f"generator_{epoch:06d}"))
    if nice:
        d = discriminator_model(n)
        d.compile(loss="binary_crossentropy", optimizer="SGD")
        if epoch is None:
            d.load_weights(os.path.join(dir, "models", "discriminator"))
        else:
            d.load_weights(os.path.join(dir, "models", f"discriminator_{epoch:06d}"))
        noise = np.random.uniform(-1, 1, (batch_size * 20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, batch_size * 20)
        index.resize((batch_size * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((batch_size,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(batch_size):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (batch_size, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = -image * 127.5 + 127.5
    if epoch is None:
        image_path = os.path.join(dir, "generated", "generated_image.png")
    else:
        image_path = os.path.join(dir, "generated", f"generated_image_{epoch:06d}.png")
    Image.fromarray(image.astype(np.uint8)).save(image_path)


def log(dir):
    log_path = os.path.join(dir, "log.hdf5")
    with h5py.File(log_path, "r") as log_file:
        epoch = int(log_file["epoch"][0])
        g_losses = log_file["g_loss"]
        d_losses = log_file["d_loss"]
        print("# g_loss, d_loss")
        for i in range(epoch):
            print(f"{i},{g_losses[i]},{d_losses[i]}")
