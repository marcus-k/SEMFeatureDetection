# -*- coding: utf-8 -*-
#
# Author: Marcus Kasdorf

import cv2
import numpy as np
import pandas as pd
from typing import Callable, Tuple, Union

from .finder import EllipseFinder


class Basic(EllipseFinder):
    """
    Very basic ellipse finder based on Canny edge detection.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    threshold_constant : float, optional (default=None)
        Constant used in the adaptive thresholding. If None, Otsu's method is used.

    show_contours : bool, optional
        Whether to show the contours of the ellipses.

    invert : bool, optional (default=False)
        Whether to invert the image.

    sort : str, optional (default=None)
        Whether to sort the ellipses in x or y direction. If None, no sorting 
        is performed.

    filter : Callable[[pd.DataFrame], pd.DataFrame], optional (default=None)
        A function that filters the DataFrame. If None, no filtering is done.

    region : Tuple[int, int], optional (default=None)
        A tuple of two tuples containing the coordinates of the upper left 
        and lower right corner of the region to be processed.
    
    """

    def __init__(
        self,
        img: np.ndarray,
        threshold_constant: float = None,
        show_contours: bool = False,
        invert: bool = False, 
        sort: str = None,
        filter: Callable[[pd.DataFrame], pd.DataFrame] = None,
        region: Union[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        **kwargs
    ) -> None:
        super().__init__(img, filter, region, **kwargs)
        self.show_contours = show_contours
        self.invert = invert
        self.threshold_constant = threshold_constant

        if sort not in ["x", "y", None]:
            raise ValueError(f'Can only sort in x or y directions. (sort = "{sort}")')
        self.sort = sort


    def preprocess(self) -> np.ndarray:
        # Crop if needed
        img = self._crop(self.img)
        
        # Convert to grayscale
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Sharpen
        # kernel = np.array([[0, -1, 0],
        #                 [-1, 5, -1],
        #                 [0, -1, 0]])
        # img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

        # Denoise
        img = cv2.GaussianBlur(img, (7, 7), 0)
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

        # Gaussian adaptive thresholding
        if self.threshold_constant is not None:
            height, width = img.shape
            s = width // 8
            s = s if s % 2 == 1 else s - 1
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, s, self.threshold_constant
            )

        # Otsu's adaptive thresholding
        else:
            ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return ~img if self.invert else img

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
        self._contours = contours

        # Find ellipses in the found contours
        for i, cont in enumerate(contours):
            # Ellipses need at least 5 points
            if len(cont) < 5:
                continue

            h = hierarchy[0, i, :]

            # Only choose contours with a parent or have no children
            # if (h[3] != -1) or (h[2] == -1):
            (x, y), (maj, min), ang = cv2.fitEllipse(cont)
            ang *= -1   # fitEllipse returns angle positive in CW direction

            # Make angles not dumb
            if ang < -90:
                ang %= 90
            if ang > 90:
                ang %= -90

            df.loc[count] = [x, y, maj, min, ang]
            count += 1

        # Add eccentricity
        super()._insert_eccentricity(df)

        # Sort order
        if self.sort is not None:
            df.sort_values(f"{self.sort} center", inplace=True, ignore_index=True)

        return df

    def draw_ellipses(self, res: pd.DataFrame) -> np.ndarray:
        """
        Draws the ellipses on the image.
        
        """
        # Read image
        img = self.img.copy()

        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.show_contours:
            # Draw contours
            img = cv2.drawContours(img, self._contours, -1, (0, 255, 0), 2)

        return super().draw_ellipses(res, img)