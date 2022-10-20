from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='geoCosiCorr3D',
    version='2.1',
    author="Saif Aati",
    author_email="saif@caltech.edu, saifaati@gmail.com",
    description="Geospatial COSI-Corr 3D",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/SaifAati/geoCosiCorr3D.git',
    python_requires='>=3.5',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=['geoCosiCorr3D',
              'geoCosiCorr3D.geoOptimization',
              'geoCosiCorr3D.geoRFM',
              'geoCosiCorr3D.geoRSM',
              'geoCosiCorr3D.geoTiePoints',
              'geoCosiCorr3D.geoOrthoResampling',
              'geoCosiCorr3D.geoImageCorrelation',
              'geoCosiCorr3D.geoThreeDD',
              'geoCosiCorr3D.geoErrorsWarning',
              'geoCosiCorr3D.lib'],
)
