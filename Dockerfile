FROM geocosicorr3d_conda_py37:latest

WORKDIR /usr/src/app/geoCosiCorr3D
RUN pwd
RUN ls

# Set the environment variables
ENV LD_LIBRARY_PATH=/usr/src/app/geoCosiCorr3D/lib/:$LD_LIBRARY_PATH
ENV PYTHONPATH=$PYTHONPATH:/usr/src/app/geoCosiCorr3D

RUN echo "source activate geoCosiCorr3D" >> ~/.bashrc
RUN pip install -e .


# Set final working directory
#WORKDIR /usr/src/app/geoCosiCorr3D/geoCosiCorr3D