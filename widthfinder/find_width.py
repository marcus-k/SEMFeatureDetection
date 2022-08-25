import os, sys
from pathlib import Path

sys.path.insert(
    0, str(Path(os.path.abspath(os.path.dirname(__file__))) / "../ellipsefinder")
)

from preprocess.rmbanner import remove_banner
from find_ellipses import get_pixel_size

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import widgets
from typing import Tuple
from itertools import combinations


def get_lines(
    img: np.ndarray, 
    dir: str = "horizontal", 
    return_pre: bool = False, 
    return_edges: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    Find long straight lines in an image. Ideally, this will be the edges that define
    the nanobeam.

    """
    # Exaggerate long straight lines
    linek = np.zeros((15, 15), dtype=np.uint8)
    linek[linek.shape[0] // 2, :] = 1
    straight = cv2.morphologyEx(img, cv2.MORPH_OPEN, linek, iterations=3)

    # Attempt denoising
    pre = cv2.GaussianBlur(straight, (7, 7), 0)
    pre = cv2.fastNlMeansDenoising(pre, None, 10, 7, 21)

    # Otsu's thresholding
    ret, pre = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Canny edge detection
    edges = cv2.Canny(pre, 50, 150, apertureSize=3)

    # Find straight line from the edges
    minLineLength = 100
    lines = cv2.HoughLinesP(
        image=edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        lines=np.array([]),
        minLineLength=minLineLength,
        maxLineGap=80,
    )

    # Copy input and convert to RGB
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Draw lines
    a, b, c = lines.shape
    for i in range(a):
        cv2.line(
            output,
            (lines[i][0][0], lines[i][0][1]),
            (lines[i][0][2], lines[i][0][3]),
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Create return values
    ret = [lines, output]
    if return_pre:
        ret.append(pre)
    if return_edges:
        ret.append(edges)

    return ret


def get_line_parameters(lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Obtains the slope and intercept from line segments.

    """
    m_arr = np.zeros(len(lines))
    b_arr = np.zeros(len(lines))
    for (i, line) in enumerate(lines):
        x1, y1, x2, y2 = line.flatten()

        # Calculate slope and intercept
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Add to list
        m_arr[i] = m
        b_arr[i] = b

    return m_arr, b_arr


def remove_outliers(m: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Using the Interquartile Range, remove outliers from the slope and intercept.

    """
    q1 = np.percentile(m, 25)
    q3 = np.percentile(m, 75)
    iqr = q3 - q1

    outlier_thresh = iqr * 1.5
    inlier_mask = (m < q3 + outlier_thresh) & (m > q1 - outlier_thresh)

    m = m[inlier_mask]
    b = b[inlier_mask]

    return m, b


def group_lines(m: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Groups lines together based on their slope and intercept using the k-means
    algorithm with k = 4. Returns a list of the 4 lines' slope and intercept.

    """
    Z = np.vstack((m, b))
    Z = np.float32(Z).T

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    ret, label, center = cv2.kmeans(
        Z, 4, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS
    )

    # Sort centers by y-intercept
    center_sorted = center[center[:, 1].argsort()]
    m = center_sorted[:, 0]
    b = center_sorted[:, 1]

    return m, b


def add_line_to_plot(
    ax: plt.Axes, img: np.ndarray, m: float, b: float, color: str = "r", label: str = ""
) -> plt.Axes:
    """
    Add a straight line to an existing matplotlib plot.

    """
    x = np.arange(0, img.shape[1])
    y = m * x + b
    return ax.plot(x, y, color=color, label=label)


class UpdateButtons:
    def __init__(
        self,
        ind: int,
        ax_up: plt.Axes,
        ax_down: plt.Axes,
        line: plt.Artist,
        m: float,
        b: float,
    ) -> None:
        self._ind = ind
        self._ax_up = ax_up
        self._ax_down = ax_down
        self._line = line
        self._m = m
        self._b = b

        self._button_up = widgets.Button(self._ax_up, f"Increase b{self._ind}")
        self._button_up.on_clicked(self.increase_b)

        self._button_down = widgets.Button(self._ax_down, f"Decrease b{self._ind}")
        self._button_down.on_clicked(self.decrease_b)

    @property
    def m(self):
        return self._m

    @property
    def b(self):
        return self._b

    @property
    def ind(self):
        return self._ind

    def redraw_line(self) -> None:
        self._line.set_ydata(self._m * self._line.get_xdata() + self._b)
        plt.draw()

    def increase_b(self, event) -> None:
        self._b += 1
        self.redraw_line()

    def decrease_b(self, event) -> None:
        self._b -= 1
        self.redraw_line()


def fine_tune_lines(
    m: np.ndarray, b: np.ndarray, img: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Opens a matplotlib figure and allows the user to fine tune the lines. Once
    the user is satisfied with the lines, the user can close the window and the
    lines will be returned.

    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.subplots_adjust(bottom=0.2)
    ax.imshow(img, cmap="gray")

    # Add the lines to the plot
    ax_lines = []
    colors = ["r", "b", "g", "c"]
    for i in range(4):
        c = colors[i]
        label = i
        (l,) = add_line_to_plot(ax, img, m[i], b[i], c, label)
        ax_lines.append(l)

    # Add the buttons to the plot
    btns_list = []
    for i in range(4):
        btn_ax_up = plt.axes([0.12 + 0.1 * i, 0.15, 0.09, 0.075])
        btn_ax_down = plt.axes([0.12 + 0.1 * i, 0.05, 0.09, 0.075])
        btns = UpdateButtons(i, btn_ax_up, btn_ax_down, ax_lines[i], m[i], b[i])
        btns_list.append(btns)

    ax.legend()
    plt.show()  # Pauses execution until the plot is closed

    final_m = np.array([btn.m for btn in btns_list])
    final_b = np.array([btn.b for btn in btns_list])

    return final_m, final_b



def find_distances(m: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    """
    Find the distance between each line and each other line.

    """
    # Use the average slope of all of the line. Since (ideally) the slopes are 
    # all basically the same, it doesn't really matter anyways.
    m = m.mean()

    # Loop through all pairs of b values and find the distances
    distances = {}
    for (i, j) in combinations(range(len(b)), 2): 
        distances[(i, j)] = abs(b[i] - b[j]) / np.sqrt(1 + m**2)

    # Convert the dictionary to a pandas dataframe
    df = pd.DataFrame()
    df["Line #1"] = [i[0] for i in distances.keys()]
    df["Line #2"] = [i[1] for i in distances.keys()]
    df["Distance (px)"] = list(distances.values())

    return df


def save_distances(filename: str, distances: pd.DataFrame, suffix: str) -> None:
    """
    Save the lines to a file.

    """
    name = str(Path(filename).with_suffix("")) + f"{suffix}.csv"
    distances.to_csv(name, index=False)


def main() -> None:
    # Load image
    filename = "../images/a90l90__q002/a90l90__q002.jpg"
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Remove banner
    no_banner, loc = remove_banner(img, invert=True)

    # Get lines and parameters
    lines, output = get_lines(no_banner)
    m, b = get_line_parameters(lines)
    m_inliers, b_inliers = remove_outliers(m, b)

    # Groups lines together and fine tune the outputted parameters
    group_m, group_b = group_lines(m_inliers, b_inliers)
    tuned_m, tuned_b = fine_tune_lines(group_m, group_b, no_banner)

    # Find the distances between each line and each other line
    distances = find_distances(tuned_m, tuned_b)

    # Scale the distances by the metadata scale value
    scale = get_pixel_size(filename, ".txt")
    distances["Distance (nm)"] = distances["Distance (px)"] * scale

    # Save the lines to a file
    save_distances(filename, distances, "_widths")
    print(distances)


if __name__ == "__main__":
    main()
