# -*- coding: utf-8 -*-
#
# Author: Marcus Kasdorf

import cv2
import numpy as np
import pandas as pd
from typing import Callable, Tuple, Union

from .finder import EllipseFinder

class EM(EllipseFinder):
    """
    Ellipse finder utilizing the machine learning Expectation-Maximization algorithm  
    for segmenting images. Canny edge detection is used to find contours.

    """

    def __init__(
        self,
        img: np.ndarray,
        n = 3,
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

        if sort not in ["x", "y", None]:
            raise ValueError(f'Can only sort in x or y directions. (sort = "{sort}")')
        self.sort = sort

        self.em = None
        self.n = n

    def train(self, img: np.ndarray, n=1) -> None:
        if self.em is None:
            self.em = cv2.ml.EM_create()

        self.em.setClustersNumber(self.n)
        self.em.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 4, 0))

        # Crop image 
        img = self._crop(img)
        # Denoise image
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

        samples = np.reshape(img, (img.shape[0]*img.shape[1], -1)).astype("float")
        for i in range(n):
            self.em.trainEM(samples)

    def predict(self, img: np.ndarray) -> np.ndarray:
        if self.em is None:
            raise ValueError("EM model not trained/initialized.")

        samples = np.reshape(img, (img.shape[0]*img.shape[1], -1)).astype("float")
        labels = np.zeros(samples.shape, "uint8")
        for i in range(samples.shape[0]):
            retval, probs = self.em.predict2(samples[i])
            labels[i] = retval[1] * (255/self.n) # make it [0,255] for imshow
        
        return np.reshape(labels, img.shape)

    def preprocess(self) -> np.ndarray:
        # Read image as grayscale and crop as needed
        # img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Crop image 
        img = self._crop(self.img)
        
        # Denoise image
        img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

        # Using EM, segment the image
        labels = self.predict(img)

        # Invert image if needed
        if self.invert:
            labels = ~labels

        # Threshold image
        labels[labels > 0] = 255

        return labels

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

        # Binarize image
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[img > 0] = 1

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

    def run(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        if self.em is None:
            raise ValueError("EM model not trained/initialized.")

        return super().run()

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