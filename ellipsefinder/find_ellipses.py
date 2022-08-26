#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Marcus Kasdorf
# 
# Various functions for processing the results of ellipse finding


import argparse
from pathlib import Path
from typing import List
from glob import glob
from operator import attrgetter
from pprint import pprint
import re

import cv2
import numpy as np
import tifffile
import pandas as pd

from methods import *
from preprocess.rmbanner import remove_banner

VERBOSE = False


def display_images_cv2(imgs: List[np.ndarray]) -> None:
    # Display images
    i = 0
    while i < len(imgs):
        img = imgs[i]

        title = f"Detected Ellipses. Total images: {len(imgs)}"
        cv2.imshow(title, img)

        # Handle image sequence and exit button
        exit_button = False
        while not exit_button:
            k = cv2.waitKeyEx(100)
            if k == ord("d"):  # Forwards: 'd' key
                break
            elif k == ord("a"):  # Backwards: 'a' key
                i -= 2
                break
            elif k == 27:
                exit_button = True

            # Check if window is still open
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                exit_button = True
        else:
            break

        i = (i + 1) % len(imgs)

    cv2.destroyAllWindows()


def save_image(filename: str, img: np.ndarray, suffix: str = "_edited") -> None:
    """
    Save image to file using OpenCV imwrite.

    Note: OpenCV imwrite saves as BGR, so we need to convert from RGB.

    """
    path = Path(filename)
    name = str(path.parent / path.stem) + suffix + path.suffix
    cv2.imwrite(name, img)

    if VERBOSE:
        print(f"saved: {name}")


def save_result(filename: str, res: pd.DataFrame, suffix: str = "_res") -> None:
    path = Path(filename)
    name = str(path.parent / path.stem) + f"{suffix}.csv"
    res.to_csv(name)

    if VERBOSE:
        print(f"{res}\n")


def get_pixel_size(filename: str, suffix: str = None) -> float:
    """
    Attempt to get the pixel size from the metadata or just from somewhere
    in the file.

    """
    # Change suffix if provided
    if suffix is not None:
        path = Path(filename)
        filename = str(path.parent / path.stem) + suffix

    # Check if file exists
    if not Path(filename).exists():
        raise FileNotFoundError(f"File not found when getting pixel size: {filename}")

    # See if pixel size is in Tiff file metadata (if it is a Tiff file)
    try:
        with tifffile.TiffFile(filename) as tif:
            try:
                # Pixel size in nm/px
                pixel_size = tif.sem_metadata["ap_pixel_size"][1]
                return pixel_size
            except KeyError:
                return 1.0
    except tifffile.TiffFileError:
        pass

    # See if pixel size is anywhere is the file (assuming it's text readable)
    with open(filename) as f:
        try:
            for line in f:
                if re.search("pixel ?size ?= ?", line, re.IGNORECASE):
                    # Pixel size in nm/px
                    pixel_size = float(line.split("=")[1])
                    return pixel_size
        except UnicodeDecodeError:
            pass

    return 1.0


def calculate_parameters(filename: str, df: pd.DataFrame, suffix: str = None) -> None:
    # Find scaling factor
    scale = get_pixel_size(filename, suffix)

    if VERBOSE:
        print(f"scale: {scale} nm/px")

    # Scale diameters
    df["x diameter (nm)"] = df["x diameter"] * scale
    df["y diameter (nm)"] = df["y diameter"] * scale

    # Approximate perimeter of ellipse (Ramanujan)
    a = df["x diameter (nm)"] / 2
    b = df["y diameter (nm)"] / 2
    df["perimeter (nm)"] = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

    # Area of ellipse
    df["area (nm^2)"] = np.pi * a * b

    # Get distances between holes
    df["distance (nm)"] = np.sqrt(
        df["x center"].diff()**2 + df["y center"].diff()**2
    ) * scale


def filter_res(res: pd.DataFrame, reset_index=True) -> pd.DataFrame:
    # cond = res["eccentricity"] < 0.3

    # Minimum diameter
    cond = res["x diameter"] >= 15
    cond &= res["y diameter"] >= 15

    # Maximum diameter
    cond &= res["x diameter"] <= 1000
    cond &= res["y diameter"] <= 1000

    res = res.query("@cond")
    return res.reset_index(drop=True) if reset_index else res


def draw_ellipses(
    res: pd.DataFrame, 
    img: np.ndarray, 
    ellipse_thickness: int = 2
) -> np.ndarray:
    """
    Default implementation of drawing the found 
    ellipses onto the original image.

    """
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
            ellipse_thickness,
        )

    return img


def parse_args() -> argparse.Namespace:
    """
    Args parser so command line testing is easier. AAMED parameter arguments
    are set up so it was easier to test with.
    
    """
    class SortingHelpFormatter(argparse.HelpFormatter):
        """
        Sort help menu options alphabetically.
        """
        def add_arguments(self, actions):
            actions = sorted(actions, key=attrgetter("option_strings"))
            super(SortingHelpFormatter, self).add_arguments(actions)

    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        prog="find_ellipses.py", formatter_class=SortingHelpFormatter
    )
    parser.add_argument("src", nargs="+", help="input images")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument(
        "-a",
        "--theta-arc",
        help="AAMED DP contour curvature threshold",
        type=float,
        default=np.pi / 3,
    )
    parser.add_argument(
        "-l",
        "--lambda-arc",
        help="AAMED DP contour length bounds",
        type=float,
        default=3.4,
    )
    parser.add_argument(
        "-T",
        "--T-val",
        help="AAMED ellipse validation threshold",
        type=float,
        default=0.77,
    )
    args = parser.parse_args()

    # Edit verbose flag
    global VERBOSE
    VERBOSE = args.verbose

    # Check for unresolved glob patterns
    for i in range(len(args.src) - 1, -1, -1):
        if any(item in args.src[i] for item in ["*", "?", "[", "]"]):
            args.src[i:i] = glob(args.src.pop(i))

    if VERBOSE:
        pprint(vars(args))
        print()

    return args


def main() -> None:
    """
    Main function to mainly test the functions and different algorithms.

    """
    import time

    start = time.time()
    args = parse_args()

    for filename in args.src:
        # Load image
        original = cv2.imread(filename)

        # Remove banner
        no_banner, loc = remove_banner(original, invert=True)
        save_image(filename, no_banner, "_nobanner")

        pre, img, res = Basic(
            no_banner,
            invert=False,
            sort="y",
            filter=filter_res,
            # region=(None, 686),
            show_contours=True,
            # _no_pre = True,
        ).run()
        # pre, img, res = AAMED(
        #     no_banner,
        #     1200,
        #     1300,
        #     args.theta_arc,
        #     args.lambda_arc,
        #     args.T_val,
        #     sort="y",
        #     filter=filter_res,
        #     region=(None, 686),
        # ).run()
        # pre, img, res = Canny(
        #     no_banner,
        #     10,
        #     20,
        #     3,  
        #     True,
        #     sort="x",
        #     filter=filter_res,
        #     region=(None, 686),
        # ).run()

        # model = EM(
        #     no_banner,
        #     n=3,
        #     invert=True,
        #     sort="y",
        #     # filter=filter_res,
        #     show_contours=True,
        # )
        # model.train(no_banner)
        # pre, img, res = model.run()

        save_image(filename, pre, "_pre")
        save_image(filename, img)
        calculate_parameters(filename, res)
        save_result(filename, res)

    if VERBOSE:
        print(f"Time elapsed: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
