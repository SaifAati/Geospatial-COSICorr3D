from setuptools import setup

if __name__ == '__main__':
    setup(name="geoCosiCorr3D",
          version="3.0.1",
          packages=[
              "geoCosiCorr3D",
              "geoCosiCorr3D.geo3DDA",
              "geoCosiCorr3D.geoCore",
              "geoCosiCorr3D.geoCore.base",
              "geoCosiCorr3D.geoErrorsWarning",
              "geoCosiCorr3D.geoImageCorrelation",
              "geoCosiCorr3D.geoOptimization",
              "geoCosiCorr3D.geoOrthoResampling",
              "geoCosiCorr3D.geoRFM",
              "geoCosiCorr3D.georoutines",
              "geoCosiCorr3D.georoutines.utils",
              "geoCosiCorr3D.geoRSM",
              "geoCosiCorr3D.geoRSM.geoRSM_metadata",
              "geoCosiCorr3D.geoTiePoints",
              "geoCosiCorr3D.utils"
          ],
          )
