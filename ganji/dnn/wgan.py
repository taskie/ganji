"""
WGAN implementation. (experimental)

Original: https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py
"""

import os
import random
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from PIL import Image
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Layer,
    Reshape,
    UpSampling2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop

import ganji.datasets
import ganji.project


class RandomWeightedAverage(Layer):
    """Provides a (random) weighted average between real and generated image samples"""

    # python - How to substitute `keras.layers.merge._Merge` in `tensorflow.keras` - Stack Overflow
    # https://stackoverflow.com/questions/58133430/how-to-substitute-keras-layers-merge-merge-in-tensorflow-keras

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = tf.random.uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class WGAN:
    def __init__(self, dir):
        self.directory = dir
        self.config, _state = ganji.project.load_metadata(dir)

        # Input shape
        self.batch_size = self.config.batch_size
        self.unit = self.config.unit
        self.img_rows = 4 * self.unit
        self.img_cols = 4 * self.unit
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=["accuracy"])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=["accuracy"])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(1024, input_dim=self.latent_dim))
        model.add(Activation("tanh"))
        model.add(Dense(128 * self.unit * self.unit))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("tanh"))
        model.add(Reshape((self.unit, self.unit, 128), input_shape=(128 * self.unit * self.unit,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(64, (5, 5), padding="same"))
        model.add(Activation("tanh"))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(self.channels, (5, 5), padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):
        model = Sequential()

        model.add(Conv2D(64, (5, 5), padding="same", input_shape=self.img_shape))
        model.add(Activation("tanh"))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (5, 5)))
        model.add(Activation("tanh"))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("tanh"))
        model.add(Dense(1))
        # model.add(Activation("sigmoid"))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def _load_state(self, *, epoch=None):
        dir = self.directory
        self.config, self.state = ganji.project.load_metadata(dir)
        if epoch is not None:
            self.generator.load_weights(os.path.join(dir, "models", f"generator_{epoch:06d}.h5"))
            self.critic.load_weights(os.path.join(dir, "models", f"critic_{epoch:06d}.h5"))
        else:
            if os.path.exists(os.path.join(dir, "models", "generator.h5")):
                self.generator.load_weights(os.path.join(dir, "models", "generator.h5"))
            if os.path.exists(os.path.join(dir, "models", "critic.h5")):
                self.critic.load_weights(os.path.join(dir, "models", "critic.h5"))

    def _load_data(self):
        dir = self.directory
        config = self.config
        path = os.path.join(dir, "dataset.npy")
        if os.path.exists(path):
            return np.load(path)
        else:
            codepoints = ganji.datasets.find_codepoints(config.codepoint_set)
            font_index = 0 if config.font_index is None else config.font_index
            thickness_quantiles = (config.thickness_quantile_min, config.thickness_quantile_max)
            if config.dataset_random_seed is not None:
                randomizer = random.Random(config.dataset_random_seed)
            else:
                randomizer = None
            data = ganji.datasets.load_data_for_gan(
                codepoints,
                config.font,
                4 * config.unit,
                font_index=font_index,
                thickness_quantiles=thickness_quantiles,
                randomizer=randomizer,
            )
            np.save(path, data)
            return data

    def train(self):
        self._load_state()

        batch_size = self.config.batch_size
        epoch_end = self.config.epoch_end
        epoch_start = self.state.epoch

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        X_train = self._load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.0
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        self.train_will_start()

        for epoch in range(epoch_start, epoch_end):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for layer in self.critic.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    layer.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print(f"{epoch:d} [D loss: {1 - d_loss[0]:f}] [G loss: {1 - g_loss[0]:f}]")

            # If at save interval => save generated image samples
            # if epoch % sample_interval == 0:
            #     self.sample_images(epoch)

            self.epoch_did_end(epoch, d_loss=d_loss, g_loss=g_loss)

    def _log_path(self):
        return os.path.join(self.directory, "log.h5")

    def train_will_start(self):
        dir = self.directory
        for subdirname in ["training", "models"]:
            subdir = os.path.join(dir, subdirname)
            if not os.path.isdir(subdir):
                os.mkdir(subdir)

        log_path = self._log_path()
        epoch_end = self.config.epoch_end

        if not os.path.exists(log_path):
            with h5py.File(log_path, "a") as f:
                f.create_dataset("epoch", (1,), dtype=np.int32)
                f.create_dataset("g_loss", (epoch_end + 1, 2), maxshape=(None, 2), dtype=np.float32)
                f.create_dataset("d_loss", (epoch_end + 1, 2), maxshape=(None, 2), dtype=np.float32)

    def epoch_will_start(self, epoch):
        self.state.epoch = epoch
        self.state.update_time = datetime.now().timestamp()

    def epoch_did_end(self, epoch, *, d_loss, g_loss):
        dir = self.directory
        log_path = self._log_path()
        epoch_end = self.config.epoch_end
        model_backup_interval = 100
        model_save_interval = 10
        image_save_interval = 1
        image_generate_seed = 0

        with h5py.File(log_path, "a") as log_file:
            log_file["epoch"][0] = epoch + 1
            log_file["d_loss"][epoch] = d_loss
            log_file["g_loss"][epoch] = g_loss

        if epoch % image_save_interval == 0 or epoch == epoch_end - 1:
            self.save_imgs(epoch, seed=image_generate_seed)

        if epoch % model_backup_interval == 0 or epoch == epoch_end - 1:
            self.generator.save_weights(os.path.join(dir, "models", f"generator_{epoch:06d}.h5"), True)
            self.critic.save_weights(os.path.join(dir, "models", f"critic_{epoch:06d}.h5"), True)

        if epoch % model_save_interval == 0 or epoch == epoch_end - 1:
            self.generator.save_weights(os.path.join(dir, "models", "generator.h5"), True)
            self.critic.save_weights(os.path.join(dir, "models", "critic.h5"), True)
            self.state.epoch = epoch + 1
            ganji.project.dump_state(dir, self.state)

    def save_imgs(self, epoch, *, rows=None, columns=None, generate_mode=False, seed=None):
        dir = self.directory
        r = 10 if rows is None else rows
        c = 10 if columns is None else columns
        n = r * c
        noise = np.random.RandomState(seed).normal(0, 1, (n, self.latent_dim))
        gen_imgs = self.generator.predict(noise, verbose=generate_mode)

        # Invert and rescale images 0 - 255
        gen_imgs = 127.5 - 127.5 * gen_imgs
        h, w = gen_imgs.shape[1:3]
        combined_image = np.zeros((r * h, c * w), dtype=gen_imgs.dtype)

        cnt = 0
        for i in range(r):
            for j in range(c):
                combined_image[i * h : (i + 1) * h, j * w : (j + 1) * w] = gen_imgs[cnt, :, :, 0]
                cnt += 1

        if generate_mode:
            image_path = os.path.join(dir, "generated", f"{epoch:06d}.png")
        else:
            image_path = os.path.join(dir, "training", f"{epoch:06d}.png")

        Image.fromarray(combined_image.astype(np.uint8)).save(image_path)

    def generate(self, *, epoch=None, rows=None, columns=None, seed=None):
        dir = self.directory
        self._load_state(epoch=epoch)

        for subdirname in ["generated"]:
            subdir = os.path.join(dir, subdirname)
            if not os.path.isdir(subdir):
                os.mkdir(subdir)

        if epoch is None:
            epoch = self.state.epoch - 1

        self.save_imgs(epoch, generate_mode=True, rows=rows, columns=columns, seed=seed)


def train(dir):
    impl = WGAN(dir)
    impl.train()


def generate(dir, *, epoch=None, rows=None, columns=None, seed=None):
    impl = WGAN(dir)
    impl.generate(epoch=epoch, rows=rows, columns=columns, seed=seed)


def log(dir):
    log_path = os.path.join(dir, "log.h5")
    with h5py.File(log_path, "r") as log_file:
        epoch = int(log_file["epoch"][0])
        g_losses = log_file["g_loss"]
        d_losses = log_file["d_loss"]
        print("epoch,g_loss,d_loss")
        for i in range(epoch):
            print(f"{i},{1-g_losses[i][0]},{1-d_losses[i][0]}")
