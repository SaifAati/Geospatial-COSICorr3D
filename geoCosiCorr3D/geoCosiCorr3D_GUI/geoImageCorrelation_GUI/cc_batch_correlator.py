import time

from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.batch_utils import BatchSelector
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.cc_correlator import *
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.gui_utils import *


class BatchCorrelator(Window):
    def __init__(self, parent, top_level):
        Window.__init__(self, parent, top_level, "Batch Correlator")

        entries = load_defaults(CORRELATION.CORR_PARAMS_CONFIG, corrParamToConfig)
        entries['Output Folder.'] = SimpleContainer("")
        entries['Output Folder..help_string'] = SimpleContainer('Path where output images will be saved.')

        # Create frame for left side of screen
        left_f = self.make_frame(padx=0, pady=0, expand=1, side='left')
        # Create batch selector
        self.batch_selector = BatchSelector(left_f, self, correlator_param, get_params)
        self.batch_selector.pack(fill="both", expand=1, padx=5, pady=5)
        # Create run button and progressbar
        self.make_run_bar(self.make_run(), self.batch_selector.params_f, 'Run correlations',
                          'Executing batch correlation', 'Batch correlation complete!', False)
        test_run = ttk.Button(self.runbar_f, text='Test Run', command=self.make_run(True))
        test_run.info_tip = TimedToolTip(test_run,
                                         'Visualize execution order without\nactually running any correlation.')
        test_run.pack(side='top')

        # Create output section
        output_f = ttk.Frame(left_f)
        output_f.pack(side='left')
        self.redirect_validation(output_f, self.batch_selector.params_f)
        create_row(output_f, 'Output Folder', [('', 'plfs')])
        self.output_var = entries['Output Folder.']

    @splitcall
    @threaded
    def make_run(self, test_run=False):
        time.sleep(0.1)  # allows main thread to not get locked from dialog callback
        for base, target in self.batch_selector.pairs():
            config = param_to_config(corrParamToConfig)

            # execute correlation
            config['base_image_path'] = base.path
            config['target_image_path'] = target.path
            config['base_band'] = base.band
            config['target_band'] = target.band
            params = config['correlator_params']
            output_folder = self.output_var.get()
            config['output_path'] = Correlate.make_output_path(output_folder, config['base_image_path'],
                                                               config['target_image_path'],
                                                               config['correlator_name'], params['window_size'][0],
                                                               params['step'][0])
            print()
            print(json.dumps(config))

            if test_run:
                time.sleep(1)
            else:
                Correlate.from_config(config)


if __name__ == '__main__':
    run(BatchCorrelator)
