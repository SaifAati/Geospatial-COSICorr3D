Copyright 2021 Saif Aati (saif@caltech.edu || saifaati@gmail.com)


# geoCosiCorr3D

Free, and open-source software tailored for satellite image processing.
geoCosiCorr3D is adept at handling various types of satellite imagery, including push-broom, frame, and push-frame sensors.
geoCoiCorr3D excels in rigorous sensor model (RSM) refinement, rational function model (RFM) refinement,
and offers advanced features like orthorectification,
sub-pixel image correlation, and 3D surface displacement extraction.
Designed for professionals and researchers in geospatial analysis,
geoCoiCorr3D bridges the gap between complex data processing needs and practical applications.

See the [NEWS](NEWS.md) for the most recent additions and upgrades.

Contact Information
-------------------

Questions, comments, and bug reports can be sent to:
[Saif Aati](mailto:saif@caltech.edu)

    - saif@caltech.edu
    - saifaati@gmail.com
# Introduction 


# Workflow 
![Alt text](Figs/WorkFlow.png?raw=true "Title")


# Installation and dependencies
--------------
To install `geoCosiCorr3D` from source:
### Option 1: 

1- Set and activate `geoCosiCorr3D` environment:

    conda env create --file geoCosiCorr3D.yml
    conda activate geoCosiCorr3D

2- Set shared libraries:

For Linux, you have to append the path to the [lib](https://github.com/SaifAati/geoCosiCorr3D/blob/main/geoCosiCorr3D/lib/) directory to LD_LIBRARY_PATH in .bashrc to be able to use geoCosiCorr3D shared libraries,  
by adding the following line: 
    
    LD_LIBRARY_PATH=~/geoCosiCorr3D/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH


Third party libraries and packages (optional):

1- Ames Stereo Pipeline ([ASP](https://github.com/NeoGeographyToolkit/StereoPipeline.git)) for WV1 and WV2 CCD correction using [wv_correct] : set the binary path of ASP in the configuration file ('geoConfig.py')


Note:

To update the `geoCosiCorr3D` env you can use the following cmd:

    conda env update --file geoCosiCorr3D.yml --prune

### Option 2: Docker
1- Build the image:

    docker-compose -f docker-compose.yml build geocosicorr3d
2- Run a container:

    docker-compose -f  docker-compose.yml run geocosicorr3d


# geoCosiCorr3D: CLI + GUI
[CLI + GUI doc](Doc/GUI_DOC.md)


# License
[License](LICENSE)

# Citation
If you are using this software for academic research or publications we ask that you please cite this software as:

<a id="1">[1]</a> Aati, S., Milliner, C., Avouac, J.-P., 2022. A new approach for 2-D and 3-D precise measurements of ground deformation from optimized registration and correlation of optical images and ICA-based filtering of image geometry artifacts. Remote Sensing of Environment 277, 113038. https://doi.org/10.1016/j.rse.2022.113038



# References

<a id="1">[2]</a> S. Leprince, S. Barbot, F. Ayoub and J. Avouac, "Automatic and Precise Orthorectification, Coregistration, and Subpixel Correlation of Satellite Images, Application to Ground Deformation Measurements," in IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 6, pp. 1529-1558, June 2007, doi: 10.1109/TGRS.2006.888937.

<a id="1">[3]</a> Aati, S.; Avouac, J.-P. Optimization of Optical Image Geometric Modeling, Application to Topography Extraction and Topographic Change Measurements Using PlanetScope and SkySat Imagery. Remote Sens. 2020, 12, 3418. https://doi.org/10.3390/rs12203418









    
