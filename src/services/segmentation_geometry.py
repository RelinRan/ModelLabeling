"""Encode disconnected mask contours as one standard YOLO polygon row."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF


def bridge_contours(parts: list[list[QPointF]]) -> list[QPointF]:
    """Join closed loops with retraced zero-area bridges, retaining every component."""
    loops = [list(part) for part in parts if len(part) >= 3]
    if not loops:
        return []
    result = loops.pop(0)
    while loops:
        vertices = np.array([(p.x(), p.y()) for p in result])
        best = None
        for loop_index, loop in enumerate(loops):
            other = np.array([(p.x(), p.y()) for p in loop])
            # Bound temporary memory for detailed ONNX contours.
            for offset in range(0, len(vertices), 256):
                distance = ((vertices[offset:offset + 256, None] - other[None]) ** 2).sum(axis=2)
                a, b = np.unravel_index(distance.argmin(), distance.shape)
                candidate = (float(distance[a, b]), loop_index, offset + int(a), int(b))
                if best is None or candidate < best:
                    best = candidate
        _, loop_index, a, b = best
        loop = loops.pop(loop_index)
        loop = loop[b:] + loop[:b]
        result = result[:a + 1] + loop + [loop[0], result[a]] + result[a + 1:]
    return result
