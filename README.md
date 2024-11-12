Copyright 2021 Saif Aati (saif@caltech.edu || saifaati@gmail.com)

[![Linux-Conda-Install (CI- GHA)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/test_conda_run_install_ubuntu.yaml/badge.svg)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/test_conda_run_install_ubuntu.yaml)
[![Linux-Docker-install (CI- GHA)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/test_docker_run_install_ubuntu.yaml/badge.svg)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/test_docker_run_install_ubuntu.yaml)
[![Correlation-conda (CI-GHA)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/geocosicorr3d_conda_env_test_correlation.yaml/badge.svg)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/geocosicorr3d_conda_env_test_correlation.yaml)
[![Ortho-conda (CI-GHA)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/geocosicorr3d_conda_env_test_ortho.yaml/badge.svg)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/geocosicorr3d_conda_env_test_ortho.yaml)

# geoCosiCorr3D

`geoCosiCorr3D` is an innovative, free and open-source software tailored for satellite image processing.
`geoCosiCorr3D` is adept at handling various types of satellite imagery, including push-broom, frame, and push-frame sensors.
At its core, `geoCoiCorr3D` excels in rigorous sensor model (RSM) refinement,
rational function model (RFM) refinement, and offers advanced processing features: including
orthorectification, sub-pixel image correlation, and 3D surface displacement extraction.

Designed for researchers in remote sensing.
`geoCosiCorr3D` serves as a critical bridge linking complex data processing requirements
with real-world applicability.
It is particularly beneficial for projects focused on change detection, time series analysis,
and applications in diverse scientific fields such as geology, geomorphology,
glaciology, planetology, as well as in the assessment and mitigation of natural disasters.

See the [NEWS](NEWS.md) for the most recent additions and upgrades.

Contact Information, Support & Contributions
-------------------
We welcome your questions, comments, and reports of any issues you encounter! Here's how you can reach out to us or
contribute to the project.
For direct inquiries or specific questions, feel free to reach out to [Saif Aati](mailto:saif@caltech.edu): (
saif@caltech.edu (Preferred) || saifaati@gmail.com)

If you encounter any problems or bugs, please report them
by [submitting an issue](https://github.com/SaifAati/Geospatial-COSICorr3D/issues)
on our GitHub project page.
This helps us track and address issues efficiently.
Your feedback and contributions are invaluable to us, and we look forward to hearing from you!

# Workflow

![Alt text](Figs/WorkFlow.png?raw=true "Title")

# Installation

There are two methods available to install `geoCosiCorr3D`: using Conda or Docker. Follow the instructions below based
on your preferred installation method.

## Option 1: Installing with Conda

1. To install using Conda, execute the following script in your terminal:
    ```bash
    ./install_cosicorr.sh --conda
    ```
   If Conda (or Miniconda) is not already installed on your machine, the script will attempt to install Miniconda
   automatically.
2. Once the installation is complete, you can activate the `geoCosiCorr3D` environment with:
    ```bash
    conda activate geoCosiCorr3D
    ```
   Alternatively, you can run Python scripts within the environment without activating it by using:
    ```bash
    conda run -n geoCosiCorr3D your_script.py
    ```
   Replace `your_script.py` with the name of the Python script you wish to run.

## Option 2: Installing with Docker

For Docker installation, execute the following command:

```bash
./install_cosicorr.sh --docker
```

This command will attempt to install Docker if it's not already installed on your system, start the Docker service, and
then pull the base `geoCosiCorr3D` Docker image. Ensure you have the necessary permissions to install software on your
system when using this option.

---

**Note:** Please follow these steps for a smooth installation process.
If you encounter any issues or need further assistance,
refer to the documentation (the documentation is still under construction 🚧) or submit an issue on the project's GitHub
page.

### Important Note on Environment Variables

⚠️ Sometimes, environment variables are not automatically picked up and set during the installation process. To ensure the software operates correctly, it is recommended to manually set these variables. For example, to set the `LD_LIBRARY_PATH` environment variable, you can use the following command in your terminal:

```bash
export LD_LIBRARY_PATH=<absolute_path_of_installation_directory>/Geospatial-COSICorr3D/lib/:$LD_LIBRARY_PATH
```
Add this line to your .bashrc or .bash_profile (depending on your shell and OS) to make the change permanent:
```bash
echo 'export LD_LIBRARY_PATH=<absolute_path_of_installation_directory>/Geospatial-COSICorr3D/lib/:$LD_LIBRARY_PATH' >> ~/.bashrc
```

# geoCosiCorr3D: CLI


The primary entry point for the `geoCosiCorr3D` command line interface (CLI) is accessible through the Python script
located at `scripts/cosicorr.py`. 
To explore the available commands and their options, you can use the `-h` or `--help` flag.
Below is a brief overview of how to use the GeoCosiCorr3D CLI:


```bash
python3 scripts/cosicorr.py -h
usage: cosicorr3d [-h] <module> ...

GeoCosiCorr3D CLI

optional arguments:
  -h, --help  show this help message and exit

modules:
  <module>
    ortho      Orthorectification
    transform  Transformation
    correlate  Correlation
```

### <span style="color:blue"> Correlation</span>

For detailed usage of the `correlate` module, execute the following command:

```bash
python3 scripts/cosicorr.py correlate -h
```

<details>
<summary>Correlate Module Usage</summary>

```bash
usage: cosicorr3d correlate [-h] [--base_band BASE_BAND] [--target_band TARGET_BAND] [--output_path OUTPUT_PATH] [--method {frequency,spatial}] [--window_size WINDOW_SIZE WINDOW_SIZE WINDOW_SIZE WINDOW_SIZE]
                            [--step STEP STEP] [--grid] [--show] [--pixel_based] [--vmin VMIN] [--vmax VMAX] [--mask_th MASK_TH] [--nb_iters NB_ITERS] [--search_range SEARCH_RANGE SEARCH_RANGE]
                            base_image target_image

positional arguments:
  base_image            Path to the base image.
  target_image          Path to the target image.

optional arguments:
  -h, --help            show this help message and exit
  --base_band BASE_BAND
                        Base image band.
  --target_band TARGET_BAND
                        Target image band.
  --output_path OUTPUT_PATH
                        Output correlation path.
  --method {frequency,spatial}
                        Correlation method to use.
  --window_size WINDOW_SIZE WINDOW_SIZE WINDOW_SIZE WINDOW_SIZE
                        Window size. (Default [64])
  --step STEP STEP      Step size. (Default [8,8])
  --grid                Use grid.
  --show                Show correlation. (Default False)
  --pixel_based         Enable pixel-based correlation.
  --vmin VMIN           Minimum value for correlation plot. (Default -1)
  --vmax VMAX           Maximum value for correlation plot. (Default 1)

Frequency method arguments:
  --mask_th MASK_TH     Mask threshold (only for frequency method).
  --nb_iters NB_ITERS   Number of iterations (only for frequency method).

Spatial method arguments:
  --search_range SEARCH_RANGE SEARCH_RANGE
                        Search range (only for spatial method).
```
Example:
```bash
python3 scripts/cosicorr.py correlate tests/test_dataset/BASE_IMG.TIF tests/test_dataset/TARGET_IMG.TIF --show  --vmin -3 --vmax 3
```
![Alt text](Figs/BASE_IMG_VS_TARGET_IMG_frequency_wz_64_step_8.png?raw=true "Title")
</details>


### <span style="color:blue">Batch Correlation</span>


The batch correlation feature allows performing correlation on multiple images in batch mode. It supports specifying lists of base and target images, with the script handling the correlation accordingly.

Comma-separated lists of base and target images can be passed, or wildcard patterns may be used to include all matching files in a directory.

### Options:
1. **Batch Correlation**
2. **Multiband Correlation** 

<details>
<summary>Batch Correlate Module Usage</summary>


**1- Serial Correlation:** 

```bash
python3 scripts/batch_correlation.py batch_correlate BASE_IMG_1.TIF,BASE_IMG_2.TIF TARGET_IMG_1.TIF,TARGET_IMG_2.TIF --output_path output/ --show --serialpython3 scripts/batch_correlation.py batch_correlate BASE_IMG_1.TIF,BASE_IMG_2.TIF "Target/*.TIF" --output_path output/ --show --all

```
**2- All Combinations Correlation:**
```bash
python3 scripts/batch_correlation.py batch_correlate BASE_IMG_1.TIF,BASE_IMG_2.TIF "Target/*.TIF" --output_path output/ --show --all
```
In these examples, the `--serial` option correlates images with the same index, while the `--all` option correlates all possible combinations of base and target images. If neither --serial nor --all is specified, the script defaults to --all.

**Note:** You can pass a comma-separated list of image paths or use a wildcard pattern like "folder/*.tif" to include all matching files in a directory.
</details>


<details>
<summary>Multiband Correlate Module Usage</summary>

The multiband correlation feature allows performing correlation between all possible bands in a given raster.
If the --band_combination option is specified, the correlation will be done between the specified bands.
```bash
python3 scripts/cosicorr.py multi_band_correlation input_img.TIF --band_combination "1,2;3,4" --output_path output/ --show
```   
</details>



### <span style="color:blue">Transform</span>
For detailed usage of the `transfrom` module, execute the following command:

```bash
python3 scripts/cosicorr.py transform -h
```

<details>
<summary>Transform Module Usage</summary>

#### Example Usage of the `transform` Command:

This section demonstrates how to use the `transform` command within the `geoCosiCorr3D` CLI to perform coordinate transformations. 
The examples show how to convert pixel coordinates to geographic coordinates (longitude, latitude, and altitude) and vice versa.

```bash
usage: cosicorr3d transform [-h] [--inv] [--dem_fn DEM_FN] x y <model_name> ...

positional arguments:
  x                list: x=cols and if with invert flag: lon
  y                list: y=lines and if with invert flag: lat

optional arguments:
  -h, --help       show this help message and exit
  --inv            Transform form ground to image space.
  --dem_fn DEM_FN  DEM file name (None)

model:
  <model_name>
    RFM            RFM model specific arguments
    RSM            RSM model specific arguments

```
#### Converting Pixel Coordinates to Geographic Coordinates

To convert pixel coordinates to geographic coordinates using a Rational Function Model (RFM), use the following command:

```bash
python3 scripts/cosicorr.py transform 0,1000 0,500 RFM tests/test_dataset/test_ortho_dataset/SP2_RPC.txt
```

**Output:**

- `lons`: [30.52895296 30.65688292]
- `lat`: [41.24090926 41.16826844]
- `alt`: [1102.49239388 1102.49239388]


#### Inverse Transformation: Converting Geographic Coordinates to Pixel Coordinates

For the inverse operation: converting geographic coordinates back to pixel coordinates, use the `--inv` flag:

```bash
python3 scripts/cosicorr.py transform 30.52895296,30.65688292 41.24090926,41.16826844 --inv RFM tests/test_dataset/test_ortho_dataset/SP2_RPC.txt
```

**Output:**

- `cols`: [9.70195697e-06 999.999998e+02]
- `lines`: [5.07104141e-06 500.000045e+02]


</details>

### <span style="color:blue">Orthorectification and model refinement</span>

For detailed usage of the `ortho` module, execute the following command:

```bash
python3 scripts/cosicorr.py ortho -h
```

<details>
<summary>Ortho Module Usage</summary>

```bash
usage: cosicorr3d ortho [-h] [--o_ortho O_ORTHO] [--corr_model CORR_MODEL] [--dem DEM] [--gsd GSD] [--resampling_method {sinc,bilinear}] [--debug] [--show] [--refine]
                        [--ref_img REF_IMG] [--gcps GCPS]
                        input_img <model_name> ...

positional arguments:
  input_img             Input file for ortho

optional arguments:
  -h, --help            show this help message and exit
  --o_ortho O_ORTHO     Output path for ortho. Defaults to the current working directory.
  --corr_model CORR_MODEL
                        Correction model path (None)
  --dem DEM             DEM path (None)
  --gsd GSD             Output file for ortho (None)
  --resampling_method {sinc,bilinear}
                        Resampling method (SINC)
  --debug
  --show
  --refine              Refine model, this require GCPs or reference imagery to collect GCPs
  --ref_img REF_IMG     Reference Ortho image (None)
  --gcps GCPS           GCPs file (None)

model:
  <model_name>
    RFM                 RFM model specific arguments
    RSM                 RSM model specific arguments
```
##### RSM model
```bash
usage: cosicorr3d ortho input_img RSM [-h] {Spot1,Spot2,Spot3,Spot4,Spot5,Spot15,Spot6,Spot7,Spot67,WV1,WV2,WV3,WV4,GE,QB,DG} rsm_fn

positional arguments:
  {Spot1,Spot2,Spot3,Spot4,Spot5,Spot15,Spot6,Spot7,Spot67,WV1,WV2,WV3,WV4,GE,QB,DG}
                        Sat-name
  rsm_fn                Specifies the path to the .xml DMP file. Additional formats are supported in GeoCosiCorr3D.pro.
```

##### RFM model
```bash
usage: cosicorr3d ortho input_img RFM [-h] rfm_fn

positional arguments:
  rfm_fn      RFM file name (.tiff or .TXT)
```

</details>

# geoCosiCorr3D: Time Series Analysis

## geoPCAIM [2]
Geospatial Principal Component Analysis-bases Inversion Method (geoPCAIM) is a statistically based approach
applied to a redundant surface displacement measurement
in order to filter out the measurement noise and extract the signal with the maximum spatio-temporal
coherence.
Horizontal displacement measurements obtained using the image correlation technique typically account 
for the surface deformation signal and several other artifacts such as decorrelation,
topography-related residuals, and stripe artifacts. 
Therefore, the main objective of this tool is to separate the deformation signal from the artifacts.
This approach can be considered as unsupervised learning,
for a **detailed description of this tool please read our research paper [2]**.

It should be noted that this approach does not imply smoothing the temporal variations of ground 
surface displacement. Instead, only local variations that do not correlate in space are filtered out.
In essence geoPCAIM approach could be used to reduce image geometry related artifacts (example Ridgecrest)
and to filter out the outliers over a time series of displacement (example Shisper).



## geoICA [1]
geoICA is a statistically based approach
applied to a redundant surface displacement measurement
in order to filter out the measurement noise and extract the signal with the maximum
spatio-temporal coherence. 
ICA decomposition is used rather than gepPCAIM in some cases because the sources of 
artifacts can be a major contribution to the data variance and could be filtered 
out by selecting only the first component of a geoPCAIM. 
The basis for the decomposition is that the deformation signal and the various sources
of artifacts can be assumed to be statistically independent sources. 
In particular, the artifacts are assumed to be statistically independent with 
the deformation signal. The deformation signal is presumed to be the same in all 
the 3-D measurements maps independent of the choice of a particular set of images. 
As a result, we expect the deformation signal to show in only one component. 
The hypothesis that the displacement signal is independent of the artifacts is
most likely to be verified in real applications. 
This assumption would be incorrect only in the unlikely case of a displacement 
signal similar with the pattern resulting from the geometric artifacts like
jitter or CCD artifacts.

![Alt text](Figs/geoICA_expl.png?raw=true "Title")


##### Upcoming Release Information

# geoCosiCorr3D: [GUI](Doc/GUI_DOC.md)

# License

[License](LICENSE)

# Citation

If you are using this software for academic research or publications we ask that you please cite this software as:

<a id="1">[1]</a> Aati, S., Milliner, C., Avouac, J.-P., 2022. A new approach for 2-D and 3-D precise measurements of
ground deformation from optimized registration and correlation of optical images and ICA-based filtering of image
geometry artifacts. Remote Sensing of Environment 277, 113038. https://doi.org/10.1016/j.rse.2022.113038


<a id="2">[2]</a> Aati, S., Avouac, J.-P., Rupnik, E., Deseilligny, M.-P., 2022.
Potential and Limitation of PlanetScope Images for 2-D and 3-D Earth Surface Monitoring with 
Example of Applications to Glaciers and Earthquakes. IEEE Transactions on Geoscience and Remote Sensing 1–1. 
https://doi.org/10.1109/TGRS.2022.3215821


# References

<a id="1">[3]</a> S. Leprince, S. Barbot, F. Ayoub and J. Avouac, "Automatic and Precise Orthorectification,
Coregistration, and Subpixel Correlation of Satellite Images, Application to Ground Deformation Measurements," in IEEE
Transactions on Geoscience and Remote Sensing, vol. 45, no. 6, pp. 1529-1558, June 2007, doi: 10.1109/TGRS.2006.888937.

<a id="1">[4]</a> Aati, S.; Avouac, J.-P. Optimization of Optical Image Geometric Modeling, Application to Topography
Extraction and Topographic Change Measurements Using PlanetScope and SkySat Imagery. Remote Sens. 2020, 12,3418. https://doi.org/10.3390/rs12203418









    
