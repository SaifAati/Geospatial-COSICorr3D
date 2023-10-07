# Using Ubuntu 20.04 as the base image
FROM ubuntu:focal

# Setting environment variables for non-interactive installation and paths
ENV DEBIAN_FRONTEND="noninteractive" \
    PATH="/root/miniconda3/bin:${PATH}"

# Update package lists and install necessary packages, then clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-utils vim tree wget curl build-essential \
    g++ gcc gfortran make imagemagick libimage-exiftool-perl \
    exiv2 proj-bin qt5-default libx11-6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory and copy necessary files
#TODO change to be from a git repository
WORKDIR /usr/src/app/geoCosiCorr3D
COPY ./setup.py ./LICENSE ./geoCosiCorr3D_DK.yml ./README.md ./setup.cfg ./pyproject.toml ./
COPY ./tests ./tests
COPY ./lib ./lib
COPY ./geoCosiCorr3D ./geoCosiCorr3D



## Download Miniconda installer, Install Miniconda, Initialize conda and set up the environment, all in a single layer
RUN wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir -p /root/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    conda init bash && \
    conda env create --file geoCosiCorr3D_DK.yml && \
    echo "source activate geoCosiCorr3D" >> ~/.bashrc && \
    export LD_LIBRARY_PATH=/usr/src/app/geoCosiCorr3D/lib/:$LD_LIBRARY_PATH


### Download Miniconda installer
#RUN wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#
### Install Miniconda
#RUN mkdir -p /root/.conda && \
#    bash Miniconda3-latest-Linux-x86_64.sh -b && \
#    rm Miniconda3-latest-Linux-x86_64.sh
#
## Initialize conda and set up the environment
#RUN conda init bash && \
#    conda env create --file geoCosiCorr3D_DK.yml && \
#    echo "source activate geoCosiCorr3D" >> ~/.bashrc
#
#
#RUN export LD_LIBRARY_PATH=/usr/src/app/geoCosiCorr3D/lib/:$LD_LIBRARY_PATH
