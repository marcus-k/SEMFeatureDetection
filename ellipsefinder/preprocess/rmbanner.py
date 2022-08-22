# -*- coding: utf-8 -*-
#
# Author: Marcus Kasdorf

import cv2
import numpy as np


def remove_banner(
    img: np.ndarray,
    thresh_init: float = 250,
    thresh_final: float = 126,
    kern_v: int = 13,
    kern_h: int = 13,
    kern: int = 5,
    alpha: float = 0.5,
    invert: bool = False,
    _debug: bool = False,
) -> np.ndarray:
    """
    Attempt to remove the banner from an image. Default settings 
    assume a white banner at the bottom of the screen.

    Altered from: https://github.com/lwang94/sem_size_analysis/blob/803251cdcab3d8304a365df9ac5879fcd9346270/experiments/3_Label_Data.ipynb
    Added connected components section to better find the banner. 

    Parameters
    ----------
    img : np.ndarray
        Image to remove the banner from.

    thresh_init : float, optional
        Initial threshold for the image. The default is 250.
    
    thresh_final : float, optional
        Final threshold for the image after eroding, dilating, and 
        adding. The default is 126.

    kern_v : int, optional
        Vertical kernel size for the erosion and dilation. The default is 13.

    kern_h : int, optional
        Horizontal kernel size for the erosion and dilation. The default is 13.

    kern : int, optional
        Kernel size for the erosion and dilation, after adding the vertical and
        horizontally eroded and dilated images together. The default is 5.
        Larger values may increase the chance that the largest connected component
        is the banner, but it may also artificially increase the height of the banner.

    alpha : float, optional
        The fractional mix of the vertical and horizontal lines in the adding step. 
        The default is 0.5 (equal weight of vertical and horizontal). Lower values
        weigh the horizontal lines more heavily, while higher values weigh the vertical
        lines more heavily.

    invert : bool, optional
        Whether to invert the image before processing. The default is False. If the
        banner is black instead of white, consider setting this to True.

    _debug : bool, optional
        Whether to return the intermediate steps of the algorithm. The default is False.

    Returns
    -------
    Tuple of np.ndarray
        The image with the removed banner and coordinates of the banner. If _debug is 
        True, the intermediate steps of the algorithm are returned as well in a tuple.

    """
    # Convert to grayscale
    if len(img.shape) != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Binarize image
    if invert:
        gray = ~gray
    ret, binary = cv2.threshold(gray, thresh_init, 255, cv2.THRESH_BINARY)

    # Create the horizontal and vertical kernels
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kern_v))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kern_h, 1))
    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kern, kern))

    # Find vertical and horizontal lines in the image
    img_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel, iterations=3)
    img_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hori_kernel, iterations=3)

    # Add the vertical and horizontal lines together
    img_final = cv2.addWeighted(img_v, alpha, img_h, 1.0 - alpha, 0.0)
    img_final = cv2.erode(~img_final, sq_kernel, iterations=2)

    ret, binary2 = cv2.threshold(img_final, thresh_final, 255, cv2.THRESH_BINARY)

    # Find the largest connected component
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(~binary2)
    ind_large = stats[1:, 4].argmax() + 1   # Index 0 is background

    # Assume the largest connected component is the banner
    # (not very accurate, but the height is usually correct which is important)
    x1, y1, xlen, ylen, area = stats[ind_large]
    x2 = x1 + xlen
    y2 = y1 + ylen
    loc = ((x1, y1), (x2, y2))

    # Crop out the banner
    out = img.copy()[:y1, :]

    if _debug:
        return out, loc, (img_v, img_h, img_final, binary2)
    else: 
        return out, loc
