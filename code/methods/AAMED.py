# -*- coding: utf-8 -*-
#
# Author: Marcus Kasdorf

import cv2
import numpy as np
import pandas as pd
from typing import Callable, Tuple, Union

from .finder import EllipseFinder
from .pyAAMED import pyAAMED


class AAMED(EllipseFinder):
    """
    Ellipse finder utilizing the Arc Adjacency Matrix-based Ellipse 
    Detection (AAMED) method by Meng et al. (2020).

    Parameters
    ----------
    img : ndarray
        Input image.

    height : int, optional (default=1000)
        Height used in the AAMED method.

    width : int, optional (default=1000)
        Width used in the AAMED method.

    theta_arc : float, optional (default=np.pi / 3)
        DP contour curvature threshold.

    lambda_arc : float, optional (default=3.3)
        DP contour length bounds.

    T_val : float, optional (default=0.77)
        Ellipse validation threshold.

    sort : str, optional (default=None)
        Whether to sort the ellipses in x or y direction. If None, no sorting 
        is performed.

    filter : Callable[[pd.DataFrame], pd.DataFrame], optional (default=None)
        A function that filters the DataFrame. If None, no filtering is done.

    region : Tuple[int, int] or Tuple[Tuple[int, int], Tuple[int, int]], optional (default=None)
        A tuple of two tuples containing the coordinates of the top left and
        bottom right corners of the region to be processed. If None, the whole
        image is processed.

    """

    def __init__(
        self,
        img: np.ndarray,
        height: int = 1000,
        width: int = 1000,
        theta_arc: float = np.pi / 3,
        lambda_arc: float = 3.3,
        T_val: float = 0.77,
        sort: str = None,
        filter: Callable[[pd.DataFrame], pd.DataFrame] = None,
        region: Union[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        **kwargs
    ) -> None:
        super().__init__(img, filter, region, **kwargs)
        self.width = width
        self.height = height
        self.theta_arc = theta_arc  # DP contour curvature threshold
        self.lambda_arc = lambda_arc  # DP contour length bounds
        self.T_val = T_val  # Ellipse validation threshold

        if sort not in ["x", "y", None]:
            raise ValueError(f'Can only sort in x or y directions. (sort = "{sort}")')
        self.sort = sort

    def preprocess(self) -> np.ndarray:
        img = self._crop(self.img)

        # Convert to grayscale if necessary
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img

    def extract(self, img: np.ndarray) -> pd.DataFrame:
        # Create AAMED instance and run ellipse extraction
        aamed = pyAAMED(self.height, self.width)
        aamed.setParameters(self.theta_arc, self.lambda_arc, self.T_val)
        res = aamed.run_AAMED(img)

        # Organize data in a DataFrame
        df = pd.DataFrame(
            res,
            columns=[
                "y center",
                "x center",
                "y diameter",
                "x diameter",
                "angle",
                "P score",
            ],
        )
        df = df[df.columns[[1, 0, 3, 2, 4, 5]]]
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

        return super().draw_ellipses(res, img)