import os
import geoCosiCorr3D.geoCore.constants as C

from importlib.metadata import version

__version__ = version("geoCosiCorr3D")

if not os.path.exists(C.SOFTWARE.WKDIR):
    os.makedirs(C.SOFTWARE.WKDIR)
