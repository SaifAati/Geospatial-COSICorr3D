"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2022
"""

from typing import Dict, Optional

import geoCosiCorr3D.geoCore.constants as C

method = None


def features(img1: str, img2: str, tp_params: Dict, output_folder: Optional[str] = None, show: bool = False):
    if output_folder is None:
        output_folder = C.SOFTWARE.WKDIR
    method = tp_params.get('method', C.TP_DETECTION_METHODS.ASIFT)
    if method == C.TP_DETECTION_METHODS.ASIFT:
        from geoCosiCorr3D.geoTiePoints.MicMacTP import cMicMacTp

        tp = cMicMacTp(ref_img_path=img1,
                       raw_img_path=img2,
                       scale_factor=tp_params.get('scale_factor', C.ASIFT_TP_PARAMS.SCALE_FACTOR),
                       o_dir=output_folder,
                       max_pts=tp_params.get('max_tps', 50),
                       plot_tps=show)
        return tp.o_tp_path

    if method == C.TP_DETECTION_METHODS.CVTP:
        # TODO: Not tested with master (dev)
        raise NotImplementedError

    if method == C.TP_DETECTION_METHODS.GEOSIFT:
        # TODO: Not tested with master (dev)
        raise NotImplementedError
    if method == C.TP_DETECTION_METHODS.ASP_IAGD:
        # TODO: Future work
        raise NotImplementedError
    if method == C.TP_DETECTION_METHODS.ASP_SIFT:
        # TODO: Future work
        raise NotImplementedError
