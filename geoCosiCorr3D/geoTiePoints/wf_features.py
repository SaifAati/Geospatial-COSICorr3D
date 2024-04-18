"""
# Author : Saif Aati
# Contact: SAIF AATI  <saif@caltech.edu> <saifaati@gmail.com>
# Copyright (C) 2024
"""
import os
from typing import Dict, Optional
from pathlib import Path
import geoCosiCorr3D.geoCore.constants as C
from geoCosiCorr3D.geoCore.geoRawTp import RawGeoTP


def features(img1: str, img2: str, tp_params: Dict, o_folder: Optional[str] = None, show: bool = False):
    if o_folder is None:
        o_folder = C.SOFTWARE.WKDIR
    method = tp_params.get('method', C.TP_DETECTION_METHODS.ASIFT)
    if method == C.TP_DETECTION_METHODS.ASIFT:
        from geoCosiCorr3D.geoTiePoints.MicMacTP import AsiftKpsMM

        tp = AsiftKpsMM(ref_img_path=img1,
                        raw_img_path=img2,
                        scale_factor=tp_params.get('scale_factor', C.ASIFT_TP_PARAMS.SCALE_FACTOR),
                        o_dir=o_folder,
                        max_pts=tp_params.get('max_tps', 50),
                        plot_tps=show)
        return tp.o_tp_path

    elif method == C.TP_DETECTION_METHODS.SIFT_CV:
        from geoCosiCorr3D.geoTiePoints.geoTP import SiftKpDistributionCv2, Cv2DescriptorMatcher

        kps1 = SiftKpDistributionCv2(img1)
        features1 = kps1()
        kps2 = SiftKpDistributionCv2(img2)
        features2 = kps2()
        match = Cv2DescriptorMatcher(features1, features2)
        matches = match()
        print(f'{C.TP_DETECTION_METHODS.SIFT_CV}: matches:{matches.shape[0]}')
        match_fn = os.path.join(o_folder,
                                f'{Path(img1).stem}_VS_{Path(img2).stem}_{C.TP_DETECTION_METHODS.SIFT_CV}_matches.pts')
        print(f'{C.TP_DETECTION_METHODS.SIFT_CV}:{match_fn}')
        match_fn = RawGeoTP.write_kp_fn(img1, img2, C.TP_DETECTION_METHODS.SIFT_CV, matches, match_fn)
        if show:
            RawGeoTP.plot_matches_v1(kps1.img, kps2.img, matches, f'{match_fn}.png')
        return match_fn

    elif method == C.TP_DETECTION_METHODS.GEOSIFT:
        # TODO: Not tested with master (dev)
        raise NotImplementedError
    elif method == C.TP_DETECTION_METHODS.ASP_IAGD:
        # TODO: Future work
        raise NotImplementedError
    elif method == C.TP_DETECTION_METHODS.ASP_SIFT:
        # TODO: Future work
        raise NotImplementedError
    else:
        raise ('Not a valid method')


if __name__ == '__main__':
    folder = '/home/saif/PycharmProjects/GEO_COSI_CORR_3D_WD/KPs_WD/'
    img1 = os.path.join(folder, "BASE_IMG.TIF")
    img2 = os.path.join(folder, "TARGET_IMG.TIF")
    tp_params = {'method': C.TP_DETECTION_METHODS.SIFT_CV}
    features(img1, img2, tp_params, show=True, o_folder=folder)
