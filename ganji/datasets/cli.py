"""Command Line Interface."""

import argparse
import random

import numpy as np

from ganji.datasets.codepoints import find_codepoints, str_to_codepoints
from ganji.datasets.font import _make_density_dict, load_bitmaps, load_data_for_gan


def _bitmap_value_to_str(x: int) -> str:
    if x == 255:
        return "%%"
    if x > 192:
        return "**"
    elif x > 64:
        return "++"
    elif x > 0:
        return "--"
    else:
        return "  "


def _bitmap_to_asciiart(bitmap: np.ndarray) -> str:
    s = ""
    for i in range(bitmap.shape[0]):
        s += "".join(_bitmap_value_to_str(x) for x in bitmap[i]) + "\n"
    return s


def main():
    parser = argparse.ArgumentParser(
        description="Build datasets from font files with FreeType."
    )
    parser.add_argument(
        "-c",
        "--codepoint-set",
        help="codepoint set (kanji|joyo-kanji|hiragana|ascii) [default: kanji]",
        default="kanji",
    )
    parser.add_argument(
        "-D",
        "--density-quantile-max",
        type=float,
        help="quantile of maximum density",
        default=None,
    )
    parser.add_argument(
        "-d",
        "--density-quantile-min",
        type=float,
        help="quantile of minimum density",
        default=None,
    )
    parser.add_argument("-F", "--font", help="font file", required=True)
    parser.add_argument(
        "-I", "--font-index", type=int, help="font index [default: 0]", default=0
    )
    parser.add_argument("-S", "--size", type=int, help="size [default: 32]", default=32)
    parser.add_argument("-s", "--characters", help="characters")
    parser.add_argument(
        "--show-density", help="show density", default=False, action="store_true"
    )
    parser.add_argument(
        "--randomize", help="randomize output", default=False, action="store_true"
    )
    args = parser.parse_args()
    if args.characters is not None:
        codepoints = str_to_codepoints(args.characters)
    else:
        codepoints = find_codepoints(args.codepoint_set)
    density_quantiles = (args.density_quantile_min, args.density_quantile_max)
    if density_quantiles[0] is None and density_quantiles[1] is None:
        density_quantiles = None

    if args.show_density:
        bitmaps = load_bitmaps(
            codepoints,
            [(args.font, args.font_index, args.size)],
            density_quantiles=density_quantiles,
        )
        density_dict = _make_density_dict(bitmaps, sizes=[args.size])
        for codepoint, density in sorted(density_dict.items(), key=lambda t: t[1]):
            print(f"{chr(codepoint)} (U+{codepoint:04X}) {density}")
        return

    if args.randomize:
        randomizer = random.Random()
    else:
        randomizer = None

    data = load_data_for_gan(
        codepoints,
        args.font,
        args.size,
        font_index=args.font_index,
        density_quantiles=density_quantiles,
        randomizer=randomizer,
    )
    for i in range(data.shape[0]):
        print(_bitmap_to_asciiart(data[i, :, :, 0]))
