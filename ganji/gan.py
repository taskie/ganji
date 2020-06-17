"""GAN implementation."""

import math
import os

import numpy as np
import tensorflow.keras as keras
from PIL import Image

import ganji.freetype_loader
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


def load_image_data(font, n, codepoints):
    return ganji.freetype_loader.load_data_for_gan(font, 4 * n, codepoints)


def train(dir):
    # prepare
    for subdirname in ["training", "models"]:
        subdir = os.path.join(dir, subdirname)
        if not os.path.isdir(subdir):
            os.mkdir(subdir)
    obj = ganji.project._load_metadata(dir)
    props = obj["props"]
    font = props["font"]
    batch_size = props["batch_size"]
    epoch_end = props["epoch_end"]
    n = props["unit"]
    codepoints = ganji.freetype_loader.find_codepoints(props["codepoint_set"])
    state = obj.get("state")
    if state is None:
        state = {"epoch": 0}
        obj["state"] = state
    epoch_start = state["epoch"]
    x_train = load_image_data(font, n, codepoints)
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
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
        print("Number of batches", int(x_train.shape[0] / batch_size))
        for index in range(int(x_train.shape[0] / batch_size)):
            # train discriminator
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            # image_batch = x_train[index*batch_size:(index+1)*batch_size]
            image_batch = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
            generated_images = g.predict(noise, verbose=0)
            if index == 0 and (epoch < 100 or epoch % 10 == 0):
                image = combine_images(generated_images)
                image = -image * 127.5 + 127.5
                image_path = os.path.join(dir, "training", f"{epoch:06d}.png")
                Image.fromarray(image.astype(np.uint8)).save(image_path)
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
        if epoch % 10 == 9:
            g.save_weights(os.path.join(dir, "models", "generator"), True)
            d.save_weights(os.path.join(dir, "models", "discriminator"), True)
            ganji.project._dump_metadata(dir, obj)
        if epoch % 100 == 99:
            e1 = epoch + 1
            g.save_weights(os.path.join(dir, "models", f"generator_{e1:06d}"), True)
            d.save_weights(os.path.join(dir, "models", f"discriminator_{e1:06d}"), True)
        state["epoch"] = epoch + 1


def generate(dir, *, epoch=None, nice=False):
    for subdirname in ["generated"]:
        subdir = os.path.join(dir, subdirname)
        if not os.path.isdir(subdir):
            os.mkdir(subdir)
    obj = ganji.project._load_metadata(dir)
    props = obj["props"]
    batch_size = props["batch_size"]
    n = props["unit"]
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
