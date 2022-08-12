# -*- coding: utf-8 -*-
#
# Author: Marcus Kasdorf

import cv2
import numpy as np
import pandas as pd
from typing import Callable, Tuple, Union

from .finder import EllipseFinder


class Canny(EllipseFinder):
    def __init__(
        self,
        img: np.ndarray,
        threshold1: float,
        threshold2: float,
        apertureSize: int = 3,
        L2gradient: bool = False,
        sort: str = None,
        filter: Callable[[pd.DataFrame], pd.DataFrame] = None,
        region: Union[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(img, filter, region, **kwargs)
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.apertureSize = apertureSize
        self.L2gradient = L2gradient

        if sort not in ["x", "y", None]:
            raise ValueError(f'Can only sort in x or y directions. (sort = "{sort}")')
        self.sort = sort

    def preprocess(self) -> np.ndarray:
        # Read image
        img = self._crop(self.img)

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold image
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply canny edge detection
        edges = cv2.Canny(
            img,
            self.threshold1,
            self.threshold2,
            apertureSize=self.apertureSize,
            L2gradient=self.L2gradient,
        )

        return edges

    def extract(self, img: np.ndarray) -> pd.DataFrame:
        # Organize data in a DataFrame
        df = pd.DataFrame(
            columns=[
                "x center",
                "y center",
                "x diameter",
                "y diameter",
                "angle",
            ],
        )
        count = 0

        # Find contours
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        # Find ellipses in the found contours
        for i, cont in enumerate(contours):
            # Ellipses need at least 5 points
            if len(cont) < 5:
                continue

            h = hierarchy[0, i, :]

            # Only choose contours with a parent or have no children
            if (h[3] != -1) or (h[2] == -1):
                (x, y), (maj, min), ang = cv2.fitEllipse(cont)
                ang *= -1  # fitEllipse returns angle positive in CW direction

                # Make angles not dumb
                if ang < -90:
                    ang %= 90
                if ang > 90:
                    ang %= -90

                # if self.min <= min <= self.max and self.min <= maj <= self.max:
                df.loc[count] = [x, y, maj, min, ang]
                count += 1

        # Add eccentricity
        super()._insert_eccentricity(df)

        # Sort order
        if self.sort is not None:
            df.sort_values(f"{self.sort} center", inplace=True, ignore_index=True)

        return df
