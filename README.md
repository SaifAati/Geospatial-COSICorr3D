Copyright 2021 Saif Aati (saif@caltech.edu || saifaati@gmail.com)

[![Linux-Conda-Install (CI- GHA)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/test_conda_run_install_ubuntu.yaml/badge.svg)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/test_conda_run_install_ubuntu.yaml)
[![Linux-Docker-install (CI- GHA)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/test_docker_run_install_ubuntu.yaml/badge.svg)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/test_docker_run_install_ubuntu.yaml)
[![Linux-Docker-install (CI- GHA)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/test_docker_run_install_ubuntu.yaml/badge.svg)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/test_docker_run_install_ubuntu.yaml)
[![Correlation-conda (CI-GHA)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/geocosicorr3d_conda_env_test_correlation.yaml/badge.svg)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/geocosicorr3d_conda_env_test_correlation.yaml)
[![Correlation-conda (CI-GHA)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/geocosicorr3d_conda_env_test_correlation.yaml/badge.svg)](https://github.com/SaifAati/Geospatial-COSICorr3D/actions/workflows/geocosicorr3d_conda_env_test_correlation.yaml)

# geoCosiCorr3D

GeoCosiCorr3D is an innovative,free and open-source software tailored for 
satellite image processing.
geoCosiCorr3D is adept at handling various types of satellite imagery,
including push-broom, frame, and push-frame sensors.
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

Contact Information
-------------------

Questions, comments, and bug reports can be sent to:
[Saif Aati](mailto:saif@caltech.edu)

    - saif@caltech.edu
    - saifaati@gmail.com


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

# geoCosiCorr3D: CLI

# geoCosiCorr3D: [GUI](Doc/GUI_DOC.md)

# License

[License](LICENSE)

# Citation

If you are using this software for academic research or publications we ask that you please cite this software as:

<a id="1">[1]</a> Aati, S., Milliner, C., Avouac, J.-P., 2022. A new approach for 2-D and 3-D precise measurements of
ground deformation from optimized registration and correlation of optical images and ICA-based filtering of image
geometry artifacts. Remote Sensing of Environment 277, 113038. https://doi.org/10.1016/j.rse.2022.113038

# References

<a id="1">[2]</a> S. Leprince, S. Barbot, F. Ayoub and J. Avouac, "Automatic and Precise Orthorectification,
Coregistration, and Subpixel Correlation of Satellite Images, Application to Ground Deformation Measurements," in IEEE
Transactions on Geoscience and Remote Sensing, vol. 45, no. 6, pp. 1529-1558, June 2007, doi: 10.1109/TGRS.2006.888937.

<a id="1">[3]</a> Aati, S.; Avouac, J.-P. Optimization of Optical Image Geometric Modeling, Application to Topography
Extraction and Topographic Change Measurements Using PlanetScope and SkySat Imagery. Remote Sens. 2020, 12,
3418. https://doi.org/10.3390/rs12203418









    
