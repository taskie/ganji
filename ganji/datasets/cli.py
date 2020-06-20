"""Command Line Interface."""

import argparse

import numpy as np

from ganji.datasets.codepoints import find_codepoints, str_to_codepoints
from ganji.datasets.font import load_data_for_gan


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
    parser = argparse.ArgumentParser(description="Build data sets from font files with FreeType.")
    parser.add_argument(
        "-c",
        "--codepoint-set",
        help="codepoint set (kanji|jouyou-kanji|hiragana|ascii) [default: kanji]",
        default="kanji",
    )
    parser.add_argument("-F", "--font", help="font file", required=True)
    parser.add_argument("-I", "--font-index", type=int, help="font index [default: 0]", default=0)
    parser.add_argument("-S", "--size", type=int, help="size [default: 32]", default=32)
    parser.add_argument("-s", "--characters", help="characters")
    parser.add_argument(
        "-T", "--thickness-quantile-max", type=float, help="quantile of maximum thickness", default=None
    )
    parser.add_argument(
        "-t", "--thickness-quantile-min", type=float, help="quantile of minimum thickness", default=None
    )
    args = parser.parse_args()
    if args.characters is not None:
        codepoints = str_to_codepoints(args.characters)
    else:
        codepoints = find_codepoints(args.codepoint_set)
    thickness_quantiles = (args.thickness_quantile_min, args.thickness_quantile_max)
    if thickness_quantiles[0] is None and thickness_quantiles[1] is None:
        thickness_quantiles = None
    data = load_data_for_gan(
        codepoints, args.font, args.size, font_index=args.font_index, thickness_quantiles=thickness_quantiles,
    )
    for i in range(data.shape[0]):
        print(_bitmap_to_asciiart(data[i, :, :, 0]))
