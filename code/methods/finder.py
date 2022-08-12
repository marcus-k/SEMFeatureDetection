#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Marcus Kasdorf

import cv2
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union
import numpy as np
import pandas as pd


class EllipseFinder(ABC):
    """
    Abstract base class for ellipse finding algorithms.
    
    """

    def __init__(
        self, 
        img: np.ndarray, 
        filter: Callable[[pd.DataFrame], pd.DataFrame] = None,
        region: Union[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        verbose: bool = False,
        _no_pre: bool = False,
        _ellipse_thick: int = 2,
    ) -> None:
        self.img = img
        self.region = region
        self.verbose = verbose
        self._no_pre = _no_pre
        self._ellipse_thick = _ellipse_thick

        if filter is not None:
            self.filter = filter
        else:
            self.filter = lambda x: x

    @abstractmethod
    def preprocess(self) -> np.ndarray:
        """
        Should return the preprocessed image ready 
        for `self.extract()` to use.

        """
        ...

    @abstractmethod
    def extract(self, img: np.ndarray) -> pd.DataFrame:
        """
        Should return a DataFrame containing the detected 
        ellipse parameters with at least the columns:

        x center, y center, x diameter, y diameter, angle, eccentricity

        See `_insert_eccentricity` for adding eccentricity.

        """
        ...

    def _crop(self, img: np.ndarray) -> np.ndarray:
        """
        Convenience function cropping the input image to the
        specified region, likely used in `self.preprocess()`.

        If the abscissa or ordinate are None in any of the 
        coordinates, it is taken to be the last index in the array.

        """
        if self.region is not None:
            region = np.asarray(self.region)
            x1, y1 = 0, 0

            if len(region.shape) > 2:
                raise ValueError(f"Invalid region shape: {self.region}")
            region = region.reshape(-1, 2)

            # Single coordinate case
            if region.shape[0] >= 1:
                if region[-1, 0] is None:
                    x2 = img.shape[1]
                else:
                    x2 = region[-1 ,0]
                if region[-1, 1] is None:
                    y2 = img.shape[0]
                else:
                    y2 = region[-1, 1]
            else:
                raise ValueError(f"Invalid region shape: {self.region}")

            # Two coordinate case:
            if region.shape[0] >= 2:
                # Starting pixel
                if region[0, 0] is None:
                    x1 = img.shape[1]
                else:
                    x1 = region[0 ,0]
                if region[0, 1] is None:
                    y1 = img.shape[0]
                else:
                    y1 = region[0, 1]

            if region.shape[0] >= 3:
                raise ValueError(f"Invalid region shape: {self.region}")
            
            return img[y1:y2, x1:x2]
        else:
            return img

    @classmethod
    def _insert_eccentricity(
        cls, 
        df: pd.DataFrame, 
        x: str = "x diameter", 
        y: str = "y diameter",
    ) -> None:
        """
        Convenience function to add the eccentricity after
        extracting the ellipses.

        """
        a = df[x] / 2
        b = df[y] / 2
        df["eccentricity"] = np.sqrt(np.abs(a ** 2 - b ** 2)) / df[[x, y]].max(1)
    

    def draw_ellipses(self, res: pd.DataFrame, img: np.ndarray = None) -> np.ndarray:
        """
        Default implementation of drawing the found 
        ellipses onto the original image.

        """
        # Get image
        if img is None:
            img = self.img.copy()

        if self.verbose:
            print(res)

        for i in res.index:
            y_center = res["y center"][i]
            x_center = res["x center"][i]
            y_diameter = res["y diameter"][i]
            x_diameter = res["x diameter"][i]

            # cv2.ellipse uses angles positive in CW direction
            angle = -res["angle"][i]

            # Skip if null values exist
            if pd.isnull([y_center, x_center, y_diameter, x_diameter, angle]).any():
                continue

            cv2.ellipse(
                img,
                ((x_center, y_center), (x_diameter, y_diameter), angle),
                (255, 0, 0),
                self._ellipse_thick,
            )

        return img

    def run(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Run the ellipse finding algorithm and return the results.
        
        """
        if not self._no_pre:
            pre = self.preprocess()
        else:
            pre = self.img.copy()
        
        res = self.extract(pre)
        res = self.filter(res)
        img = self.draw_ellipses(res)

        return pre, img, res