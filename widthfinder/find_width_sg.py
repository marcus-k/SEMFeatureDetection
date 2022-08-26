from find_width import *

import PySimpleGUI as sg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tabulate import tabulate

from matplotlib.figure import Figure
from tkinter import Canvas
from typing import Tuple


def draw_figure(canvas: Canvas, figure: Figure) -> FigureCanvasTkAgg:
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def find_lines(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attempt to find the four lines that outline the nanobeam.
    
    """
    # Get lines and parameters
    lines, output = get_lines(img)
    m, b = get_line_parameters(lines)
    m_inliers, b_inliers = remove_outliers(m, b)

    # Groups lines together and fine tune the outputted parameters
    group_m, group_b = group_lines(m_inliers, b_inliers)
    
    return group_m, group_b


def read_image(filename: str, banner: str = None) -> np.ndarray:
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    invert = None
    if banner == "White":
        invert = False
    elif banner == "Black":
        invert = True
    
    if invert is not None:
        no_banner, loc = remove_banner(img, invert=invert)
        return no_banner
    else:
        return img


def create_button(text: str, key: str) -> sg.Button:
    size = (10, 1)
    pad = ((0, 0), 3)
    font = 'Helvetica 14'
    return sg.Button(text, key=key, size=size, pad=pad, font=font, disabled=True)


def update_output(
    fig_agg: FigureCanvasTkAgg, 
    output_elem: sg.Output, 
    m: np.ndarray, 
    b: np.ndarray
) -> None:
    distances = find_distances(m ,b)
    output_elem.update(tabulate(
        distances, 
        headers = "keys", 
        showindex = False,
    ))
    fig_agg.draw()


def main():
    filename = "../images/a90l90__q002/a90l90__q002.jpg"

    # Define window
    col1 = [
        [sg.Canvas(size=(1280, 720), key="canvas")],
        [create_button(f"Decrease b{i}", f"decrease_b{i}") for i in range(4)],
        [create_button(f"Increase b{i}", f"increase_b{i}") for i in range(4)],
    ]
    col2 = [
        [sg.Text("Settings", font="Helvetica 14")],
        [
            sg.Column([
                [sg.Text("Source Image:", font="Helvetica 10")],
                [sg.Input(key="input_image_filename", expand_x=True)],
                [sg.FileBrowse(key="input_image", target=(-1, 0))],
                [
                    sg.Text("Banner Color:", pad=((0, 0), 3), font='Helvetica 10'),
                    sg.Combo(["White", "Black", "None"], key="banner", default_value="White", readonly=True, enable_events=True),
                ],
                [
                    sg.Checkbox("Transpose", key="transpose", default=False, enable_events=True),
                ],
            ], element_justification="left", expand_x=True)
        ],
        [sg.Button("Calculate", key="calculate", font="Helvetica 14")],
        [sg.Text("Output:", pad=((0, 0), 3), font='Helvetica 14')],
        [sg.Text("", key="output", font="Consolas 12", background_color="white", text_color="black", expand_x=True)],
        
    ]
    layout = [[
        sg.Column(col1, element_justification="center"),
        sg.Column(col2, element_justification="center", vertical_alignment="top", expand_x=True)
    ]]
    window = sg.Window("Width Detection", layout, finalize=True, resizable=True)

    # Get elements
    canvas_elem = window["canvas"]
    banner_elem = window["banner"]
    transpose_elem = window["transpose"]
    output_elem = window["output"]
    canvas = canvas_elem.TKCanvas

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 8))
    fig_agg = draw_figure(canvas, fig)
    ax_lines, m, b = None, None, None
    colors = ["r", "b", "g", "c"]
    active_plot = False

    # Main event loop
    while True:
        event, values = window.read()
        if event is None:
            break

        if event != "__TIMEOUT__":
            print("Events:", event)
            print("Values:", values)
            print()

        if event == "calculate":
            # Display image
            filename = values["input_image_filename"]
            img = read_image(filename, banner=banner_elem.get())
            if transpose_elem.get():
                img = img.T
            ax.cla()
            ax_im = ax.imshow(img, cmap="gray")

            # Add beam edge lines to the plot
            m, b = find_lines(img)
            ax_lines = []
            for i in range(len(m)):
                c = colors[i]
                label = i
                (l,) = add_line_to_plot(ax, img, m[i], b[i], c, label)
                ax_lines.append(l)
            ax.legend()

            # Update output
            update_output(fig_agg, output_elem, m, b)
            active_plot = True

            # Enable buttons
            if len(m) >= 4 and len(b) >= 4:
                for i in range(4):
                    window[f"increase_b{i}"].update(disabled=False)
                    window[f"decrease_b{i}"].update(disabled=False)
            else:
                for i in range(4):
                    window[f"increase_b{i}"].update(disabled=True)
                    window[f"decrease_b{i}"].update(disabled=True)
        
        # Update line parameters when buttons are pressed
        if active_plot and "increase_b" in event:
            i = int(event[-1])
            b[i] += 1
            ax_lines[i].set_ydata(m[i] * ax_lines[i].get_xdata() + b[i])
            fig_agg.draw()
            update_output(fig_agg, output_elem, m, b)

        if active_plot and "decrease_b" in event:
            i = int(event[-1])
            b[i] -= 1
            ax_lines[i].set_ydata(m[i] * ax_lines[i].get_xdata() + b[i])
            fig_agg.draw()
            update_output(fig_agg, output_elem, m, b)

    window.close()


if __name__ == "__main__":
    main()