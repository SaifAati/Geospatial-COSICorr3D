from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.gui_utils import *
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.tk_utils import run
from geoCosiCorr3D.geoImageCorrelation.correlate import Correlate
from geoCosiCorr3D.geoCore.constants import CORRELATION

#see param_to_config() in gui_utils
corrParamToConfig= {
    "Base Image.": "base_image_path",
    "Target Image.": "target_image_path", 
    "Base Image.Band": "base_band", 
    "Target Image.Band": "target_band", 
    "Correlation File.": "output_path", 
    "Correlator": "correlator_name", 
    "Initial Window Size.X,Initial Window Size.Y,Final Window Size.X,Final Window Size.Y": "correlator_params.window_size", 
    "Step.X,Step.Y": "correlator_params.step", 
    "&Gridded Output": "correlator_params.grid", 
    "Mask Threshold": "correlator_params.mask_th", 
    "Robustness Iterations": "correlator_params.nb_iters", 
    "Search Range.X,Search Range.Y": "correlator_params.search_range"
}

def get_params(correlator):
    """Returns the constructor for the parameter window."""
    if correlator == "frequency":
        return FrequencyCorrelator
    elif correlator == "spatial":
        return SpatialCorrelator
    else: raise Exception("Correlator must be 'frequency' or 'spatial'") 

correlator_param = ("Correlator", "dfrequency/spatial")

#Windows
class Correlator(Window):
    def __init__(self, parent, top_level):
        Window.__init__(self, parent, top_level, "Correlator")

        load_defaults(CORRELATION.CORR_PARAMS_CONFIG, corrParamToConfig)

        input_f  = self.make_frame(pady=(1, 5), text='Input')
        output_f = self.make_frame(pady=(1, 5), text="Output")
        
        self.make_run_bar(self.run, input_f, 'Run correlation', 'Executing correlation', 'Correlation complete!')

        create_row(input_f, "Base Image"  , [("", "plt"), ("Band", "PBase Image.")])
        create_row(input_f, "Target Image", [("", "plt"), ("Band", "PTarget Image.")])

        params = lambda: self.new_window(get_params(read('Correlator')))
        create_row(input_f, None, [correlator_param, ("Params", params)])

        create_row(output_f, "Correlation File", [("", "ps ")])

    @threaded
    def run(self):
        config = param_to_config(corrParamToConfig)

        Correlate.from_config(config)
        
    
class FrequencyCorrelator(Window):
    def __init__(self, parent, top_level, entry_prefix=None):
        Window.__init__(self, parent, top_level, "Frequency Correlator")

        self.load_template("Parameters")
        create_rows(self.params_f, [("Initial Window Size", [("X", "iw128"), ("Y", "iw128")]),
                                   ("Final Window Size"  , [("X", "iw32" ), ("Y", "iw32" )]),
                                   ("Step"               , [("X", "ip8"  ), ("Y", "ip8"  )]),
                                   (None                 , [("Robustness Iterations", "in3")]),
                                   (None                 , [("Mask Threshold", "fu0.9"   )]),
                                   (None                 , [("Gridded Output", "c1"      )])],
                                   entry_prefix=entry_prefix)


class SpatialCorrelator(Window):
    def __init__(self, parent, top_level, entry_prefix=None):
        Window.__init__(self, parent, top_level, "Spatial Correlator")

        self.load_template("Parameters")
        create_rows(self.params_f, [("Initial Window Size", [("X", "iw128"), ("Y", "iw128")]),
                                  ("Final Window Size"   , [("X", "iw32" ), ("Y", "iw32" )]),
                                  ("Step"                , [("X", "ip8"  ), ("Y", "ip8"  )]),
                                  ("Search Range"        , [("X", "ip10" ), ("Y", "ip10" )]),
                                  (None                  , [("Gridded Output", "c1"      )])],
                                  entry_prefix=entry_prefix)


if __name__ == '__main__':
    run(Correlator)