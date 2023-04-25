FROM ubuntu:20.04
ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

RUN apt-get install -y vim tree wget

WORKDIR /usr/src/app/geoCosiCorr3D
COPY ./*.py ./
COPY ./LICENSE ./
COPY geoCosiCorr3D ./geoCosiCorr3D
COPY ./geoCosiCorr3D_DK.yml ./
COPY ./README.md ./

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN mkdir /root/.conda
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:${PATH}"
RUN conda init bash
RUN conda env create --file geoCosiCorr3D_DK.yml
RUN echo "source activate geoCosiCorr3D" >> ~/.bashrc
RUN apt-get update --fix-missing
RUN apt-get install -y --no-install-recommends build-essential curl g++ gcc gfortran
RUN apt-get install -y make imagemagick libimage-exiftool-perl exiv2 proj-bin qt5-default
RUN apt-get install -y libx11-6

RUN LD_LIBRARY_PATH=/usr/src/app/geoCosiCorr3D/geoCosiCorr3D/libs/:$LD_LIBRARY_PATH
RUN export LD_LIBRARY_PATH
RUN cp -r -p /usr/src/app/geoCosiCorr3D/geoCosiCorr3D/lib/* /usr/lib/
WORKDIR /usr/src/app/geoCosiCorr3D/geoCosiCorr3D
