# ImageJ Steps

This was the process used to generate the masks and the regions of interest (ROIs) from the SEM images using ImageJ.

## Creating new masks and ROIs

Add all the holes as ROIs
1. `File > Open` the desired image.
2. `Analyze > Tools > ROI Manager...` to open the ROI manager tool.
3. Using the ellipse (or other) selection tool from the toolbar, draw a ROI around the desired hole.
4. Click `Add [t]` in the ROI manager to add your selection to the ROI list.
5. Repeat steps 3 & 4 for each hole.

Save the ROIs
1. Once all the desired ROIs are in the ROI list, make sure none of them are highlighted by clicking `deselect` in the ROI manager.
2. Click `More > Save...` in the ROI manager and save your ROIs. It should try to save a `.zip` file containing the `.roi` ROI file for each selection. If it's saving a `.roi` file, likely one of the ROIs is highlighted.

Create a binary mask from the ROIs
1. With all the ROIs in the ROI list, highlight all of them by `CTRL+Click` or `CTRL+A`.
2. Right-click the selection and `OR (Combine)` the ROIs.
3. `Edit > Selection > Create Mask` to open a new window with the created binary mask.
4. With the binary image in focue, `File > Save As` and select a desired file type to save in. Usually TIFF files are appropriate.

Create formatted CSV of hole dimensions
1. Open the detailed list of ROIs in the ROI manager using `More > List`.
2. Save the detailed list using `File > Save As...` in new window that popped up to save a preliminary CSV file.
3. In the terminal, navigate to the folder where you saved the CSV and run `python format_roi_csv.py [filename.csv]` to format the data. Add the arguments `-s [x, y]` to sort in a direction or `-m [metadata.txt]` to read metadata from another file.

## Editing existing ROIs

Update ROIs
1. `File > Open` the desired image.
2. In the ROI Manager, `More > Open...` the ROI zip file.
3. To edit an existing ROI, highlight it in the ROI manager. Then, adjust your selection as desired. Once finished, click `Update` in the ROI manager. Save in the same way as before.