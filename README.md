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
poetry run ganji new -F /usr/share/fonts/OTF/ipaexm.ttf -d 0.025 -D 0.975 ipaexm
# or
cd ipaexm
poetry run ganji init -F /usr/share/fonts/OTF/ipaexm.ttf -d 0.025 -D 0.975
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

### Show density of glyphs

```sh
poetry run python -m ganji.datasets \
    -F /usr/share/fonts/OTF/ipaexm.ttf -I 0 -S 40 \
    -c joyo-kanji -d 0.995 --show-density
```

```txt
嚇 (U+5687) 77.52
翻 (U+7FFB) 78.29
鋼 (U+92FC) 78.97625
繊 (U+7E4A) 79.169375
酬 (U+916C) 79.875625
綱 (U+7DB1) 79.9275
醜 (U+919C) 80.4075
鬱 (U+9B31) 80.62625
醸 (U+91B8) 81.4575
欄 (U+6B04) 81.908125
臓 (U+81D3) 83.02
```

### Render glyphs

```sh
poetry run python -m ganji.datasets \
    -F /usr/share/fonts/OTF/ipaexm.ttf -I 0 -S 16 \
    -c hiragana -d 1
```

```txt
                        ++**++
  --                  --++--++
  ++++    ----        --++--++
  --**--  ++********++--++**++
  --**            **
  ++++            **
  ++--  --++------**++****--
  ++--    --********++----
  ++              **
--++--++          **--
--++++--      ----**--
  **++    ++**********++++
  ++++  --**------**--++**++
  ++++    ++******++    --

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
