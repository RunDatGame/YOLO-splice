"""视频帧预处理：对比度增强与锐化。"""

from __future__ import annotations

import cv2
import numpy as np


class FramePreprocessor:
    """增强管节环形/接缝结构的可视特征。"""

    def __init__(
        self,
        clahe_clip: float = 2.0,
        clahe_grid: tuple[int, int] = (8, 8),
        sharpen: bool = True,
        denoise: bool = False,
    ) -> None:
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        self.sharpen = sharpen
        self.denoise = denoise
        self._sharpen_kernel = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        if self.denoise:
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = self.clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a_channel, b_channel])
        result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        if self.sharpen:
            result = cv2.filter2D(result, -1, self._sharpen_kernel)

        return result
