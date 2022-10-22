## geoImageCorrelation (sub-package of geoCosiCorr3D)

### Co-registration of Optically Sensed Images and Correlation

<!--[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

TODO: uncomment this and remove one # from each header-->

**COSI-Corr** is a software developed at the California Institute of Technology for the accurate geometrical processing of optical satellite and aerial imagery. The software allows precise co-registration of time-series of images and sub-pixel measurement of ground surface deformation.

This software is packaged as a set of tools which can be used together or independantly. Each python script whose name begins with `cc_` is a tool which can be executed. All GUI tools can be accessed through the [hub window](#hub-window). Information on each tool can be found below.

*Note: When in doubt, try hovering your mouse over a field or button you are confused about. A helpful tooltip will probably appear!*

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#setup">Setup</a></li>
    <li><a href="#hub-window">Hub window</a></li>
    <li><a href="#image-viewer">Image Viewer</a></li>
    <li><a href="#correlation">Correlation</a>
        <ul>
            <li><a href="#correlation-cli">Command Line Interface</a></li>
            <li><a href="#correlation-gui">Graphical Interface</a></li>
            <li><a href="#batch-correlator">Batch Correlator</a></li>
        </ul>
    </li>
    <li><a href="../LICENSE">License</a></li>
  </ol>
</details>

# Setup

TODO: add setup details

Install conda.

Create and activate conda environment from geoImageCorrelation.yml.

export LD_LIBRARY_PATH=/insert/path/to/geoImageCorrelation/libs/:$LD_LIBRARY_PATH

# Hub Window

<p align="center">
    <img src=assets/hub.png width="250">
</p>

**Entry-point:** Run `python geoImageCorrelation/corr_gui/cc_gui.py`.

This window serves as an entry-point for all other graphical tools in this package (no CLI access). Simply click on the tool button you would wish to use. 

Alternatively, use the embedded Image Hub to view images. See [Image Viewer](#image-viewer) for details.

# Image Viewer

![image_viewer](assets/image_viewer.png)

The image-viewer allows the user to open and preview images. 

**Entry-point:** Run `python geoImageCorrelation/corr_gui/cc_viewer.py` and open an image.

### Usage

Open any images you would like to view, and you will be presented with the following windows:

<p align="center">
    <img src=assets/image_viewer_overview.png width="350">
</p>

Windows:
- Image Viewer (main window)
    - Movement tools apply to this window.
    - If this window closes, the others will too.
- Preview
    - A preview of the complete image. Click and drag to move the main window viewing panel.
- Subview
    - A zoomed in view of the main window. Has pan/zoom controls like the main window.

The toolbar at the top of the screen shows available options:
1. Undo/Redo view changes.
1. Save current view as image.
1. Select a tool:
    - Move tool: Left click and drag to pan. Right click and drag to zoom. Middle click and drag to move the subview viewing panel.
    - Box zoom tool: Draw a box. The main window will zoom to fit that box.
1. Select a band and color map:
    - All available bands of the current image can be selected from the dropdown.
    - The color map options are pulled from matplotlib color maps.

### Notes

While the _Linked_ box is checked, moving any of the opened images will move to the corresponding location on all the other images (as shown in the image above). This is accomplished using georeferencing data stored in the image files.

If the _Simplified_ box is checked, only the main Image Viewer window will be opened.

Closing the ImageHub will close all other windows.

# Correlation

This tool generates a displacement map between two overlapping orthorectified images.

There are three ways to interface with this tool: 

1) [CLI](#correlation-cli) - Command line interface for correlating two images.
1) [GUI](#correlation-gui) - Graphical interface for correlating two images.
1) [Batch](#batch-correlator) - Graphical interface for running multiple correlations with multiple images.

## Correlation CLI

Command line tool to perform correlation between two images.

**Entry-point:** Run `python geoImageCorrelation/corr_cli/cc_cli.py correlate -help`

Running this command with the `-help` option should provide all the information you need to get started. That being said, here are some tips:

### Usage

The standard usage for this command is `python cc_cli.py correlate BASE_IMAGE TARGET_IMAGE [OPTIONS]`. Here are some example calls:

Keep in mind that most options have aliases (like `-o` -> `-output` or `-c` -> `-correlator`).

```
python cc_cli.py correlate Samples/DG_Sample/BASE_IMG.TIF Samples/DG_Sample/TARGET_IMG.TIF
```
```
python cc_cli.py correlate b.tif t.tif -base_band 3 -o test/output.tiff 
```

```
python cc_cli.py correlate b.tif t.tif -c spatial -window_sizes 64 64 32 32 
```


Alternatively, you can simply pass in the path to a configuration file:

```
python cc_cli.py correlate path/to/config.json
```

Here are some example configuration json files. Options not present in the config will be filled in with their default value. For more detail on each parameter, run `python cc_cli.py correlate -help`

All-options frequency correlator config.json:
```json
{ 
    "base_image_path": "./base.tif",
    "target_image_path": "./target.tif",
    "base_band": 1,
    "target_band": 1,
    "output_path": "",
    "correlator_name": "frequency",
    "correlator_params": {
        "window_size": [64, 64, 64, 64],
        "step": [8, 8],
        "grid": true,
        "mask_th": 0.9,
        "nb_iters": 4
    }
}
```

All-options spatial correlator config.json:
```json
{
    "base_image_path": "./base.tif",
    "target_image_path": "./target.tif",
    "base_band": 1,
    "target_band": 1,
    "output_path": "",
    "correlator_name": "spatial",
    "correlator_params": {
        "window_size": [64, 64, 64, 64],
        "step": [8, 8],
        "grid": true,
        "search_range": [10, 10]
    }
}
```

## Correlation GUI

![correlator_gui](assets/correlator_gui.png)

Graphical tool to perform correlation between two images.

**Entry-point:** Run `python geoImageCorrelation/corr_gui/cc_correlator.py`

### Usage

Information on specific parameters can be found by hovering the mouse over the corresponding input for a description. 

When you have finished inputting the parameters, simply press run and wait for the message telling you correlation is complete! Detailed progress and error logs can be found in the terminal which is running the gui.

### Notes

If an input is ever invalid (like step x in the image above), the text will go red and hovering will reveal a tooltip describing what you did wrong.

If the output file is not specified, the default name and input paths will be used.

## Batch Correlator

![batch_correlator](assets/batch_correlator.png)

Graphical tool to perform multiple correlations between multiple images.

**Entry-point:** Run `python geoImageCorrelation/corr_gui/cc_batch_correlator.py`

### Usage

The main canvas contains two columns of image band slots, with a column of arrows in the middle. To add a new image, simply press Load in the corresponding column and each band of the added image will be placed into a new slot. To add a new parameter configuration to specific correlations, click on the arrows.

Slot controls:
- Left click: Drag slots within or between the columns. 
- Right click: Duplicate slot and drop the duplicate wherever the mouse is released.
- Double-click: Open an [Image Viewer](#image-viewer) with the specified image band.

Output folder can be selected in the bottom left. Correlation options are determined by the color-coded parameter box. As shown in the image above, clicking on arrows will cycle their color and a correspondingly colored parameter box will appear at the bottom. See [Correlation Gui](#correlation-gui) for details on specific parameters.

Correlation order and parameter selection can be visualized by pressing the Test Run button. This will not execute any correlation but will visually scan through the Run order, highlighting the base image, target image, and parameter box.

Once the specified parameters have been entered, press Run to begin the correlation process.

Depending on which batch option was selected in the top left, correlation will proceed as follows:

1. One to One
    - Each left slot (base image) is paired with the adjacent right slot (target image) for correlation.
    - That row's parameter color will be used.
2. One to Many
    - Each left slot (base image) is sequentially paired with each right slot (target image) for correlation.
    - The parameter color of the left slot will be used.
3. Many to Many
    - Each slot will be correlated with each other slot, regardless of column.
    - This process will be repeated with every parameter color present.



### Notes

After Run is pressed, the slots currently being correlated will be highlighted.

If there are an imbalanced number of slots filled with the _One\_to\_One_ batch type, the excess will be ignored.

Many to Many batch type starts from the upper left corner and works down the left column before working down the right column, with first slot (base image)
and later slot(target image).
