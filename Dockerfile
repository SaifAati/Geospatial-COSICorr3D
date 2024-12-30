ARG BASE_IMAGE=ghcr.io/saifaati/geospatial-cosicorr3d/base_cosicorr3d_image:base.1.2
FROM $BASE_IMAGE as builder


ENV DEBIAN_FRONTEND="noninteractive"

LABEL maintainer="Saif Aati <saifaati@gmail.com>"
USER root
RUN date

WORKDIR /usr/src/app/geoCosiCorr3D
COPY ./*.py ./
COPY ./LICENSE ./
COPY ./README.md ./
COPY ./NEWS.md ./
COPY ./setup.cfg ./
COPY scripts ./scripts
COPY tests ./tests
COPY lib ./lib
COPY geoCosiCorr3D ./geoCosiCorr3D
COPY ./run_cosicorr_tests.sh ./
RUN pip install -e .

RUN cp -r -p /usr/src/app/geoCosiCorr3D/lib/* /usr/lib/
# Set the environment variables
ENV LD_LIBRARY_PATH=/usr/src/app/geoCosiCorr3D/lib/:$LD_LIBRARY_PATH
ENV PYTHONPATH=$PYTHONPATH:/usr/src/app/geoCosiCorr3D

WORKDIR /usr/src/app/geoCosiCorr3D