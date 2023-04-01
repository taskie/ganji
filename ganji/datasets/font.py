"""Build datasets from font files with FreeType."""

from typing import Dict, List, Optional, Tuple

import freetype
import numpy as np

try:
    import PIL
except ModuleNotFoundError:
    pass


def _calc_copy_nd(
    src: Tuple[int, ...], dst: Tuple[int, ...]
) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]:
    if len(src) != len(dst):
        raise Exception("invalid dimension")
    src_result: List[Tuple[int, int]] = []
    dst_result: List[Tuple[int, int]] = []
    for i in range(len(src)):
        src_x = src[i]
        dst_x = dst[i]
        if dst_x > src_x:
            begin = (dst_x - src_x) // 2
            end = begin + src_x
            src_result.append((0, src_x))
            dst_result.append((begin, end))
        else:
            begin = (src_x - dst_x) // 2
            end = begin + dst_x
            src_result.append((begin, end))
            dst_result.append((0, dst_x))
    return (tuple(src_result), tuple(dst_result))


def _copy_2d(src: np.ndarray, dst: np.ndarray):
    s, d = _calc_copy_nd(src.shape, dst.shape)
    dst[d[0][0] : d[0][1], d[1][0] : d[1][1]] = src[
        s[0][0] : s[0][1], s[1][0] : s[1][1]
    ]


def _setup_face(font_path: str, size: int, *, index: int = 0) -> freetype.Face:
    face = freetype.Face(font_path, index)
    face.set_char_size(size * 64)
    return face


def _make_glyph_bitmap(
    face: freetype.Face,
    codepoint: int,
    *,
    size: Optional[int] = None,
    padding: bool = False,
) -> np.ndarray:
    char_index = face.get_char_index(codepoint)
    if char_index == 0:
        return None
    face.load_glyph(char_index)
    bitmap = face.glyph.bitmap
    length = len(bitmap.buffer)
    if length == 0:
        return None
    row_count = bitmap.rows
    col_count = length // row_count
    glyph_size = (row_count, col_count)
    glyph_bitmap = np.reshape(np.array(bitmap.buffer, dtype=np.uint8), glyph_size)
    if padding and size is not None:
        row_offset = (size - row_count) // 2
        col_offset = (size - col_count) // 2
        pad_width = (
            (row_offset, size - row_count - row_offset),
            (col_offset, size - col_count - col_offset),
        )
        glyph_bitmap = np.pad(glyph_bitmap, pad_width)
    return glyph_bitmap


def _make_glyph_bitmap_dict(
    face: freetype.Face,
    codepoints: List[int],
    *,
    size: Optional[int] = None,
    padding: bool = False,
) -> Dict[int, np.ndarray]:
    result = {}
    for codepoint in codepoints:
        try:
            glyph_bitmap = _make_glyph_bitmap(
                face, codepoint, size=size, padding=padding
            )
            if glyph_bitmap is None:
                continue
            result[codepoint] = glyph_bitmap
        except freetype.FT_Exception:
            pass
    return result


def _normalize_range(
    t: Tuple[Optional[float], Optional[float]], begin_min: float, end_max: float
) -> Tuple[float, float]:
    begin = begin_min if t[0] is None else t[0]
    end = end_max if t[1] is None else t[1]
    return (max(begin, begin_min), min(end, end_max))


def _make_density_dict(
    bitmaps_dict: Dict[int, List[np.ndarray]],
    sizes: List[int],
    *,
    density_indices: Optional[List[int]] = None,
) -> Dict[int, float]:
    # bitmaps_dict[codepoint][face_index]: shape=(size, size)
    if density_indices is None:
        density_indices = range(len(sizes))
    density_indices = frozenset(density_indices)
    density_dict: Dict[int, float] = {}
    for codepoint, bitmaps in bitmaps_dict.items():
        for i, bitmap in enumerate(bitmaps):
            if i not in density_indices:
                continue
            if codepoint not in density_dict:
                density_dict[codepoint] = 0.0
            density_dict[codepoint] += np.sum(bitmap) / (sizes[i] ** 2)
    density_dict = {k: v / len(density_indices) for k, v in density_dict.items()}
    return density_dict


def _calc_range_from_quantiles(
    values: List[float], quantiles: Tuple[Optional[float], Optional[float]]
) -> Tuple[float, float]:
    q_min, q_max = _normalize_range(quantiles, 0.0, 1.0)
    v_min = np.quantile(values, q_min)
    v_max = np.quantile(values, q_max)
    return (v_min, v_max)


def _filter_bitmaps_dict(
    bitmaps_dict: Dict[int, List[np.ndarray]],
    sizes: List[int],
    *,
    density_indices: Optional[List[int]] = None,
    density_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    density_quantiles: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Dict[int, List[np.ndarray]]:
    # bitmaps_dict[codepoint][face_index]: shape=(size, size)
    d = bitmaps_dict
    if density_quantiles is not None:
        if density_quantiles[0] is None and density_quantiles[1] is None:
            density_quantiles = None
    if density_range is not None or density_quantiles is not None:
        density_dict = _make_density_dict(
            bitmaps_dict, sizes, density_indices=density_indices
        )
        if density_quantiles is not None and density_range is None:
            density_range = _calc_range_from_quantiles(
                list(density_dict.values()), density_quantiles
            )
        if density_range is not None:
            density_min, density_max = _normalize_range(density_range, 0, float("+inf"))
            d_new = {}
            for codepoint, glyph_bitmaps in d.items():
                if density_min <= density_dict[codepoint] <= density_max:
                    d_new[codepoint] = glyph_bitmaps
            d = d_new
    return d


def load_bitmaps(
    codepoints: List[int],
    fonts: List[Tuple[str, int, int]],
    *,
    density_indices: Optional[List[int]] = None,
    padding: bool = False,
    density_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    density_quantiles: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Dict[int, List[np.ndarray]]:
    # fonts[i]: (path, font_index, size)
    # returns:
    glyph_bitmaps_dict: Dict[int, List[np.ndarray]] = {}
    for i, (path, font_index, size) in enumerate(fonts):
        face = _setup_face(path, size, index=font_index)
        glyph_bitmap_dict = _make_glyph_bitmap_dict(
            face, codepoints, size=size, padding=padding
        )
        for codepoint, glyph_bitmap in glyph_bitmap_dict.items():
            glyph_bitmaps = glyph_bitmaps_dict.get(codepoint)
            if glyph_bitmaps is None:
                glyph_bitmaps = []
                glyph_bitmaps_dict[codepoint] = glyph_bitmaps
            glyph_bitmaps.append(glyph_bitmap)
    for codepoint, glyph_bitmaps in glyph_bitmaps_dict.items():
        if len(glyph_bitmaps) != len(fonts):
            del glyph_bitmaps_dict[codepoint]
    glyph_bitmaps_dict = _filter_bitmaps_dict(
        glyph_bitmaps_dict,
        sizes=[f[2] for f in fonts],
        density_indices=density_indices,
        density_range=density_range,
        density_quantiles=density_quantiles,
    )
    return glyph_bitmaps_dict


def load_as_images(
    fonts: List[Tuple[str, int, int]],
    codepoints: List[int],
    *,
    density_indices: Optional[List[int]] = None,
    density_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    density_quantiles: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> List[Tuple["PIL.Image", int, int]]:
    glyph_bitmaps_dict = load_bitmaps(
        codepoints,
        fonts,
        padding=True,
        density_indices=density_indices,
        density_range=density_range,
        density_quantiles=density_quantiles,
    )
    return [
        (PIL.Image.fromarray(glyph_bitmap), i, codepoint)
        for codepoint, glyph_bitmaps in sorted(glyph_bitmaps_dict.items())
        for i, glyph_bitmap in enumerate(glyph_bitmaps)
    ]
