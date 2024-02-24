ARG BASE_IMAGE=geocosicorr3d:base
FROM $BASE_IMAGE as builder


ENV DEBIAN_FRONTEND="noninteractive"

LABEL maintainer="Saif Aati <saifaati@gmail.com>"
USER root
RUN date

WORKDIR /usr/src/app/geoCosiCorr3D
COPY ./*.py ./
COPY ./LICENSE ./
COPY ./geoCosiCorr3D_DK.yml ./
COPY ./README.md ./
COPY ./NEWS.md ./
COPY ./setup.cfg ./
COPY scripts ./scripts
COPY tests ./tests
COPY geoCosiCorr3D ./geoCosiCorr3D
RUN pip install -e .

RUN cp -r -p /usr/src/app/geoCosiCorr3D/geoCosiCorr3D/lib/* /usr/lib/
# Set the environment variables
ENV LD_LIBRARY_PATH=/usr/src/app/geoCosiCorr3D/geoCosiCorr3D/lib/:$LD_LIBRARY_PATH
ENV PYTHONPATH=$PYTHONPATH:/usr/src/app/geoCosiCorr3D


#RUN LD_LIBRARY_PATH=/usr/src/app/geoCosiCorr3D/geoCosiCorr3D/libs/:$LD_LIBRARY_PATH
#RUN export LD_LIBRARY_PATH
WORKDIR /usr/src/app/geoCosiCorr3D