from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.tkrioplt import *
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.gui_utils import Window
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.cc_viewer import ImageHub
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.cc_correlator import Correlator
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.cc_batch_correlator import BatchCorrelator
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.tk_utils import run

class geoCOSICORR3D(Window):
    def __init__(self, parent, top_level):
        Window.__init__(self, parent, top_level, "Hub")

        title = ttk.Label(self.root, text='COSI-Corr', relief='ridge', borderwidth=1, anchor='center')
        title.grid(row=0, column=0, columnspan=2, ipadx=15, ipady=15, padx=5, pady=(5, 10), sticky='news')
        
        self.row = 1

        self.make_button('Correlator'      , Correlator     )
        self.make_button('Batch Correlator', BatchCorrelator)
        

        #Embed an ImageHub
        hub_f = self.make_frame(text='Image Hub', row=1, column=1, rowspan=self.row)
        hub = self.embed_window(hub_f, ImageHub)

    def make_button(self, text, constructor):
        command = lambda: self.new_window(constructor)

        button = ttk.Button(self.root, text=text, command=command)
        button.grid(row=self.row, column=0, pady=(10, 5), padx=(15, 10), sticky='we')
        self.row += 1
        return button




if __name__ == '__main__':
    run(geoCOSICORR3D)