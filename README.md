![ipaexm (epoch=50000)](examples/dcgan/ipaexm/training/050000.png)

---

# GANJI

This kanji does not exist.

## Installation

This project uses [Poetry](https://python-poetry.org/).

### Linux (using local TensorFlow)

```bash
git clone https://github.com/taskie/ganji.git
cd ganji
# change the global setting if you want to create .venv under the project directory
# poetry config virtualenvs.in-project true
poetry install
poetry install -E tensorflow
```

### Linux (using system TensorFlow)

```bash
git clone https://github.com/taskie/ganji.git
cd ganji
poetry config virtualenvs.in-project true  # change the global setting
python3 -m venv --system-site-packages .venv
poetry install
```

### Windows

```bat
chcp 65001
git clone https://github.com/taskie/ganji.git
cd ganji
REM (change the global setting if you want to create .venv under the project directory)
REM poetry config virtualenvs.in-project true
poetry install
REM (if you want to install TensorFlow in .venv)
poetry install -E tensorflow
REM (freetype.dll may be needed under %PATH%. see https://pypi.org/project/freetype-py/)
```

## Usage

### Initialize

```sh
poetry run ganji new -F /usr/share/fonts/OTF/ipaexm.ttf -t 0.025 -T 0.975 ipaexm
# or
cd ipaexm
poetry run ganji init -F /usr/share/fonts/OTF/ipaexm.ttf -t 0.025 -T 0.975
```

### Train

```sh
cd ipaexm
poetry run ganji train
# or
poetry run ganji -C ipaexm train
```

### Generate

```sh
poetry run ganji generate
```

### Show logs

```sh
poetry run ganji log
```

## Examples

* [examples/dcgan/ipaexm](examples/dcgan/ipaexm)

## Usage (detail)

### Render glyphs

```sh
FONT=/usr/share/fonts/OTF/ipaexm.ttf
INDEX=0
poetry run python -m ganji.datasets -F "$FONT" -I "$INDEX" -c hiragana
```

### Show thickness of glyphs

```sh
poetry run python -m ganji.datasets -F "$FONT" -I "$INDEX" -c joyo-kanji -t 0.995 --show-thickness
```

```txt
蹴 (U+8E74) 77.5576171875
縄 (U+7E04) 77.6787109375
嚇 (U+5687) 77.8447265625
醸 (U+91B8) 79.09765625
魔 (U+9B54) 79.2236328125
鬱 (U+9B31) 79.3857421875
酬 (U+916C) 80.564453125
欄 (U+6B04) 80.720703125
繊 (U+7E4A) 81.025390625
臓 (U+81D3) 82.5966796875
醜 (U+919C) 83.662109375
```

## Tested Environments

* CPU: AMD Ryzen 7 1700 Eight-Core Processor
* GPU: NVIDIA GeForce GTX 1060 6GB
* RAM: SanMax Technologies DDR4-2400 16GB (2x8GB)
* Arch Linux ([`linux`](https://www.archlinux.org/packages/core/x86_64/linux/) 5.7.4.arch1-1)
    * Python 3.8.3
    * CUDA 10.2.89
    * cuDNN 7.6.5
    * [`python-tensorflow-opt-cuda`](https://www.archlinux.org/packages/community/x86_64/python-tensorflow-opt-cuda/) 2.2.0-1
* Windows 10 Home
    * Python 3.7.7
    * CUDA 10.1.243
    * cuDNN 7.6.5
* Python Libraries: see [pyproject.toml](pyproject.toml) or [poetry.lock](poetry.lock)

## References

* [eriklindernoren/Keras-GAN: Keras implementations of Generative Adversarial Networks.](https://github.com/eriklindernoren/Keras-GAN)
* [\[1511.06434\] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
* [jacobgil/keras-dcgan: Keras implementation of Deep Convolutional Generative Adversarial Networks](https://github.com/jacobgil/keras-dcgan)
* [はじめてのGAN](https://elix-tech.github.io/ja/2017/02/06/gan.html)
* [Keras 2 で”はじめてのGAN” - Qiita](https://qiita.com/IntenF/items/94da17a8931e1f14b6e3)
