# GANJI

This kanji does not exist.

## Installation

This project uses [Poetry](https://python-poetry.org/).

```bat
git clone https://github.com/taskie/ganji.git
cd ganji
REM (if you want to create .venv under the project directory)
REM poetry config virtualenvs.in-project true
poetry install
REM (if you want to install TensorFlow in .venv)
poetry install -E tensorflow
REM (if you are using Windows, freetype.dll may be needed under %PATH% .
REM  see https://pypi.org/project/freetype-py/ )
```

## Usage

### Initialize

```bat
poetry run ganji new --font c:\WINDOWS\Fonts\MSMINCHO.TTC msmincho
```

### Train

```bat
poetry run ganji train msmincho
```

### Generate

```bat
poetry run ganji generate --nice msmincho
```

## Tested Environments

* Windows 10 Home
* NVIDIA GeForce GTX 1060 6GB
* CUDA 10.1.243
* cuDNN 7.6.5

## References

* [\[1511.06434\] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
* [jacobgil/keras-dcgan: Keras implementation of Deep Convolutional Generative Adversarial Networks](https://github.com/jacobgil/keras-dcgan)
* [eriklindernoren/Keras-GAN: Keras implementations of Generative Adversarial Networks.](https://github.com/eriklindernoren/Keras-GAN)
* [はじめてのGAN](https://elix-tech.github.io/ja/2017/02/06/gan.html)
* [Keras 2 で”はじめてのGAN” - Qiita](https://qiita.com/IntenF/items/94da17a8931e1f14b6e3)
