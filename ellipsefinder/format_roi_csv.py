from operator import attrgetter
import pandas as pd
import argparse
from pathlib import Path

from find_ellipses import calculate_parameters, save_result


def parse_args() -> argparse.Namespace:
    class SortingHelpFormatter(argparse.HelpFormatter):
        """
        Sort help menu options alphabetically.
        """
        def add_arguments(self, actions):
            actions = sorted(actions, key=attrgetter("option_strings"))
            super(SortingHelpFormatter, self).add_arguments(actions)

    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        prog="format_roi_csv.py", formatter_class=SortingHelpFormatter
    )
    parser.add_argument("src", type=str, help="input roi csv from ImageJ")
    parser.add_argument("-s", "--sort", type=str, choices=["x", "y", None], help="sort by x or y")
    parser.add_argument("-m", "--metadata", type=str, help="metadata txt file")
    args = parser.parse_args()

    return args


def main():
    # Parse arguments
    args = parse_args()

    # Get paths
    file_path = Path(args.src)
    if args.metadata:
        meta_path = Path(args.metadata)
    else:
        meta_path = Path(args.src)

    # Create dataframes
    rois = pd.read_csv(file_path)
    out = pd.DataFrame(
        columns = [
            "x center",
            "y center",
            "x diameter",
            "y diameter",
            "angle",
        ],
    )

    # Populate dataframe
    out["x center"] = rois["X"] + rois["Width"] / 2
    out["y center"] = rois["Y"] + rois["Height"] / 2
    out["x diameter"] = rois["Width"]
    out["y diameter"] = rois["Height"]
    out["angle"] = pd.Series(0.0, index=out.index)

    # Sort order
    if args.sort is not None:
        out.sort_values(f"{args.sort} center", inplace=True, ignore_index=True)

    # Add parameters
    calculate_parameters(meta_path, out)    

    # Save results
    save_result(file_path, out, "_res")

if __name__ == "__main__":
    main()
