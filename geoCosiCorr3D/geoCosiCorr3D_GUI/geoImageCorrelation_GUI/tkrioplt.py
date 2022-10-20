import os

import rasterio as rio
import rasterio.plot as rplt
import tkinter as tk
import tkinter.messagebox as tkmsg
import tkinter.filedialog as tkfile
from tkinter import BooleanVar, ttk
import matplotlib as mpl
from matplotlib import backend_tools
from matplotlib.figure import Figure
from matplotlib.backend_bases import NavigationToolbar2, key_press_handler, _Mode, MouseButton
from matplotlib.backends._backend_tk import logging, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import namedtuple

import geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.settings as settings
from geoCosiCorr3D.geoImageCorrelation.geoCorr_utils import get_bands, project_path, setdefaultattr, splitcall
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.tk_utils import (Window, ToolTip, TimedToolTip)

_log = logging.getLogger(__name__)


class ImageHub(Window):
    def __init__(self, parent, top_level):
        Window.__init__(self, parent, top_level, "Image Hub", multichild=True)

        open_b = ttk.Button(self.root, text='Open', command=self.open)
        open_b.pack(padx=7)

        self.is_linked = BooleanVar(value=settings.hub_linked)
        linked = ttk.Checkbutton(self.root, text="Linked", variable=self.is_linked)
        linked.info_tip = TimedToolTip(linked,
                                       text='If checked, the view of all images will\nbe synced by geographical positioning.')
        linked.pack()

        self.simplified = BooleanVar(value=settings.hub_simplified)
        full_view = ttk.Checkbutton(self.root, text="Simplified", variable=self.simplified)
        full_view.info_tip = TimedToolTip(full_view,
                                          text='If unchecked, the image will be opened with a Preview and Subview.')
        full_view.pack()

        # self.new_window(ImageViewer, r"/home/rhalamed/geoImageCorrelation/geoImageCorrelation/geoImageCorrelation_Test/Sample/BASE_IMG.TIF", colors)
        # self.new_window(ImageViewer, r"/home/rhalamed/geoImageCorrelation/geoImageCorrelation/geoImageCorrelation_Test/Sample/BASE_IMG_VS_TARGET_IMG_frequency_wz_64_step_8.tif", colors)

    def open(self):
        paths = tkfile.askopenfilenames()
        for path in paths:
            if not path: continue

            if not get_bands(path):
                tkmsg.showerror('Invalid', f'The selected file must be a valid image: {os.path.basename(path)}')
                continue

            self.new_viewer(path)

    def new_viewer(self, path, band_name=None):
        self.new_window(ImageViewer, path, band_name=band_name, simplified=self.simplified.get())


def create_ax(fig):
    ax = fig.add_subplot(111)

    ax.set(title="", xticks=[], yticks=[])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout(pad=0)
    fig.set_facecolor('#f5f6f7')

    return ax


_PanInfo = namedtuple("_PanInfo", "button axes cid")


# Image viewing window with linking controls
class ImageViewer(Window):
    def __init__(self, parent, top_level, path, band_name=None, simplified=False):
        Window.__init__(self, parent, top_level, os.path.basename(path), multichild=True)

        self.path = path
        self.bands = get_bands(self.path)

        if band_name != None:
            self.band_provided = True
        else:
            self.band_provided = False
            band_name = list(self.bands.keys())[0]
        self.starting_band_name = band_name

        self.fig = Figure(dpi=settings.viewer_size)  # self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = create_ax(self.fig)

        axes = [self.ax]

        if simplified:
            self.preview = self.subview = None
        else:
            self.preview = self.new_window(ImagePreview, on_close=self.unregister_preview)
            self.subview = self.new_window(ImageSubview, on_close=self.unregister_subview)
            axes.extend([self.preview.ax, self.subview.ax])
        self.top_level.bind('<Configure>', self.move_locked)

        self.draw(self.bands[band_name], axes, settings.cmaps[0])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.mpl_connect("key_press_event", lambda e: key_press_handler(e, self.canvas, self.toolbar))
        self.canvas.get_tk_widget().bind("<Configure>", self.resize_event, add='+')

        # do not rename, the bar requires being named toolbar
        self.toolbar = CosiCorNavBar(self.canvas, self, not simplified, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=8)

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.draw()
        if not simplified:
            self.toolbar.set_preview_getter(lambda: self.preview)
            self.toolbar.set_subview_getter(lambda: self.subview)
            self.subview.set_zoom()

            self.preview.canvas.draw()
            self.subview.canvas.draw()
            # self.subview.register_moved()

    def draw(self, band, axes, color):
        with rio.open(self.path) as src_plot:
            for axis in axes:
                rplt.show((src_plot, band), ax=axis, cmap=color)
        # plt.close()

    def unregister_preview(self):
        self.preview = None

    def unregister_subview(self):
        self.subview = None

    def move_locked(self, e):
        locked = [window.top_level for window in [self.preview, self.subview] if window and window.locked]

        parent_top = self.top_level
        x = parent_top.winfo_x()
        y = parent_top.winfo_y() + parent_top.winfo_height()
        for window in locked:
            window.geometry(f'+{x}+{y}')
            x += window.winfo_width()
        self.resize_event(e)

    def resize_event(self, e):
        self.ax._set_view(fit_aspect(self.ax._get_view(), self.canvas))

        if hasattr(self, 'subview') and self.subview:
            self.subview.register_moved()
        self.toolbar.register_moved()


class ImagePreview(Window):
    """A full-image preview window with drag controls that allow it to control it's parent ImageViewer"""

    def __init__(self, parent, top_level):
        Window.__init__(self, parent, top_level, "Preview")
        self.locked = True

        fig = Figure(dpi=settings.preview_size)
        self.ax = create_ax(fig)

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.canvas.mpl_connect('button_press_event', self.press_pan)

        self.offset = None

    # adapted from NavigationToolbar2
    def press_pan(self, event):
        """Callback for mouse button press."""
        bar = self.parent.toolbar

        self.id_drag = self.canvas.mpl_connect("motion_notify_event", self.drag_pan)
        self.id_release = self.canvas.mpl_connect('button_release_event', self.release_pan)
        self.panning = True

        self.offset, bounds = get_bounds(self.preview_locals, event.x, event.y)
        view = convert_bounds(True, self.canvas, *bounds)

        CosiCorNavBar.current_view = view
        bar = self.parent.toolbar
        bar.run_linked(True, False, on_self=True)

    # adapted from NavigationToolbar2
    def drag_pan(self, event):
        """Callback for dragging in pan/zoom mode."""
        if not self.panning: return
        bar = self.parent.toolbar

        self.offset, bounds = get_bounds(self.preview_locals, event.x, event.y, offset=self.offset)
        view = convert_bounds(True, self.canvas, *bounds)

        CosiCorNavBar.current_view = view
        bar.run_linked(True, False, on_self=True)

    # adapted from NavigationToolbar2
    def release_pan(self, event):
        """Callback for mouse button release in pan/zoom mode."""
        if not self.panning: return
        bar = self.parent.toolbar
        self.canvas.mpl_disconnect(self.id_drag)
        self.canvas.mpl_disconnect(self.id_release)

        bar.run_linked(False, True, on_self=True)


class ImageSubview(Window):
    current_view = None

    """A zoomed-in preview window with pan controls that allow it to be controlled by the parent ImageViewer"""

    def __init__(self, parent, top_level):
        Window.__init__(self, parent, top_level, "Subview")
        self.locked = True

        fig = Figure(dpi=settings.subview_size)
        self.ax = create_ax(fig)

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.set_cursor(backend_tools.Cursors.MOVE)

        self.canvas.mpl_connect('button_press_event', self.press_pan)

    # adapted from NavigationToolbar2
    def press_pan(self, event):
        """Callback for mouse button press."""
        bar = self.parent.toolbar

        id_drag = self.canvas.mpl_connect("motion_notify_event", self.drag_pan)
        self.id_release = self.canvas.mpl_connect('button_release_event', self.release_pan)

        axes = self.canvas.figure.get_axes()
        for ax in axes:
            ax.start_pan(event.x, event.y, event.button)

        self._pan_info = _PanInfo(
            button=event.button, axes=axes, cid=id_drag)
        self.register_moved()

    # adapted from NavigationToolbar2
    def drag_pan(self, event):
        """Callback for dragging in pan/zoom mode."""
        for ax in self._pan_info.axes:
            ax.drag_pan(self._pan_info.button, event.key, event.x, event.y)
        self.register_moved()
        self.run_linked(ax._get_view())

    # adapted from NavigationToolbar2
    def release_pan(self, event):
        """Callback for mouse button release in pan/zoom mode."""
        if self._pan_info is None: return
        bar = self.parent.toolbar

        self.canvas.mpl_disconnect(self.id_release)
        self.canvas.mpl_disconnect(self._pan_info.cid)
        for ax in self._pan_info.axes:
            ax.end_pan()
        self.canvas.draw_idle()
        self._pan_info = None

    def set_zoom(self):
        """zoom in so the window is looking at the square center of the frame with 
        1/10 the full image (uses width or height depending on whats larger)"""
        if ImageSubview.current_view:
            self.ax._set_view(ImageSubview.current_view)
            self.register_moved()
            return

        scale = 30
        # get bounds and dimensions in local coordinates
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        x0, x1, y0, y1 = convert_bounds(False, self.canvas, x0, x1, y0, y1)
        width, height = x1 - x0, y1 - y0

        # make square
        if height < width:
            x0 += (width - height) // 2
            width = height
        else:
            y0 += (height - width) // 2
            height = width
        # calculate scaled bounds
        nwidth, nheight = width // scale, height // scale
        view = x0, x1, y0, y1
        view = scale_bounds(view, nwidth, nheight)

        view = fit_aspect(view, self.canvas)
        view = convert_bounds(True, self.canvas, *view)
        self.run_linked(view, True)

    def run_linked(self, view, on_self=False):
        ImageSubview.current_view = view

        if on_self:
            self.ax._set_view(fit_aspect(view, self.canvas))
            self.register_moved()

        for toolbar in self.parent.toolbar.get_toolbars():
            if toolbar == self.parent.toolbar: continue
            if not hasattr(toolbar.image_viewer, 'subview'): return
            subview = toolbar.image_viewer.subview
            if not subview: return

            subview.ax._set_view(fit_aspect(view, subview.canvas))
            subview.register_moved()

    def register_moved(self):
        viewer = self.parent
        toolbar = viewer.toolbar

        axis = self.canvas.figure.get_axes()[0]
        x0, x1 = axis.get_xlim()
        y0, y1 = axis.get_ylim()

        self.canvas.draw_idle()

        toolbar.draw_bounds(viewer, x0, y0, x1, y1)


def get_bounds(locals, event_x, event_y, offset=None):
    # find bounds and width and height
    x0, x1, y0, y1 = locals  # self.preview_locals
    width, height = x1 - x0, y1 - y0
    # calculate click offset
    if not offset:
        # if click inside box
        if x0 <= event_x <= x1 and y0 <= event_y <= y1:
            offset = x0 - event_x, y0 - event_y
        else:
            offset = -width // 2, -height // 2
    ox, oy = offset
    # calculate output bounds
    x0, x1, y0, y1 = event_x, event_x + width, event_y, event_y + height
    # add offset to output bounds
    x0, x1, y0, y1 = x0 + ox, x1 + ox, y0 + oy, y1 + oy

    return (offset, (x0, x1, y0, y1))


def convert_bounds(inverted, canvas, x0, x1, y0, y1):
    """Converts the inputted set of points into global or local coords for the specified canvas.

    Args:
        inverted (bool): True: local->global. False: global->local.
    """
    if inverted:
        transformer = canvas.figure.get_axes()[0].transData.inverted()  # transformer from local to global
    else:
        transformer = canvas.figure.get_axes()[0].transData  # transformer from global to local
    (x0, y0), (x1, y1) = transformer.transform([(x0, y0), (x1, y1)])
    (x0, x1), (y0, y1) = tuple(sorted((x0, x1))), tuple(sorted((y0, y1)))
    return x0, x1, y0, y1


def scale_bounds(view, nwidth, nheight):
    """Scales a view from the old width and height to a new width and height staying centered"""
    x0, x1, y0, y1 = view
    width, height = x1 - x0, y1 - y0

    x0, y0 = x0 + (width - nwidth) // 2, y0 + (height - nheight) // 2
    x1, y1 = x0 + nwidth, y0 + nheight

    return x0, x1, y0, y1


def fit_aspect(view, canvas):
    """Expands a view to fit perfectly with the given canvas"""
    # calculate current and target bounds
    t_width, t_height = canvas._tkcanvas.winfo_width(), canvas._tkcanvas.winfo_height()

    setdefaultattr(canvas, 'prev_width', t_width)
    setdefaultattr(canvas, 'prev_height', t_height)
    expanding = t_width - canvas.prev_width + t_height - canvas.prev_height > 0
    canvas.prev_width, canvas.prev_height = t_width, t_height

    x0, x1, y0, y1 = view
    width, height = x1 - x0, y1 - y0

    # calculate current and target aspect ratios
    ratio = width / height
    t_ratio = t_width / t_height

    n_width, n_height = width, height
    # needs to scale horizontally
    if (ratio < t_ratio and expanding) or (ratio >= t_ratio and not expanding):
        n_width = t_ratio * height
    # needs to scale vertically
    else:
        n_height = width / t_ratio

    view = scale_bounds(view, n_width, n_height)

    return view


# Adapted from NavigationToolbar2Tk from matplotlib.backends
class CosiCorNavBar(NavigationToolbar2, ttk.Frame):
    """
    Summary
    ----------
    This class acts as a toolbar that controls the canvas provided. 
    It should be called toolbar, and its parent should have an is_linked property.
    The canvas should already have axes added to it before the constructor is called.

    Attributes
    ----------
    canvas : `FigureCanvas`
        The figure canvas on which to operate.
    win : tk.Window
        The tk.Window which owns this toolbar.
    pack_toolbar : bool, default: True
        If True, add the toolbar to the parent's pack manager's packing list
        during initialization with ``side='bottom'`` and ``fill='x'``.
        If you want to use the toolbar with a different layout manager, use
        ``pack_toolbar=False``.
    """

    current_view = None

    # list of toolitems to add to the toolbar, format is:
    # (
    #   text, # the text of the button (often not visible to users)
    #   tooltip_text, # the tooltip shown on hover (where possible)
    #   image_file, # name of the image for the button (without the extension)
    #   name_of_method, # name of the method in NavigationToolbar2 to call
    # )
    toolitems = (
        # ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
        (None, None, None, None),
        ('Pan',
         'Left button pans, Right button zooms,\n'
         'Middle button moves subview,\n'
         'x/y fixes axis, CTRL fixes aspect',
         'move', 'pan'),
        ('Zoom', 'Zoom to rectangle\nx/y fixes axis', 'zoom_to_rect', 'zoom'),
        (None, None, None, None),
        # ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'), #TODO turn into settings window with dpi slider and linked option
    )

    def __init__(self, canvas, image_viewer, zoom, *, pack_toolbar=True):
        # Avoid using self.window (prefer self.canvas.get_tk_widget().master),
        # so that Tool implementations can reuse the methods.
        self.window = image_viewer.root
        self.image_viewer = image_viewer

        ttk.Frame.__init__(self, master=self.window, borderwidth=2,
                           width=int(canvas.figure.bbox.width), height=50)

        self._buttons = {}
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                # Add a spacer; return value is unused.
                self._Spacer()
            else:
                self._buttons[text] = button = self._Button(
                    text,
                    # str(mpl.cbook._get_data_path(f"images/{image_file}.png")),
                    project_path(f"./geoImageCorrelation_GUI/assets/icons/{image_file}.png"),
                    toggle=callback in ["zoom", "pan"],
                    command=getattr(self, callback),
                )
                if tooltip_text is not None:
                    ToolTip(button, tooltip_text)

        self._label_font = tk.font.Font(root=self.window, size=10)

        # This filler item ensures the toolbar is always at least two text
        # lines high. Otherwise the canvas gets redrawn as the mouse hovers
        # over images because those use two-line messages which resize the
        # toolbar.
        label = ttk.Label(master=self, font=self._label_font,
                          text='\N{NO-BREAK SPACE}\n\N{NO-BREAK SPACE}')
        label.pack(side=tk.RIGHT)

        self.message = tk.StringVar(master=self)
        self._message_label = tk.Label(master=self, font=self._label_font,
                                       textvariable=self.message)
        self._message_label.pack(side=tk.RIGHT)

        NavigationToolbar2.__init__(self, canvas)
        if pack_toolbar:
            self.pack(side=tk.BOTTOM, fill=tk.X)

        # self._id_drag_test = self.canvas.mpl_connect('motion_notify_event', self.test_mouse_move)
        self.panning = False
        self.offset = None
        self.pan()

        if all([not isinstance(x, ImageViewer) for x in self.image_viewer.parent.children]):
            CosiCorNavBar.current_view = None

        self.band_var = tk.StringVar()
        band_names = list(self.image_viewer.bands.keys())
        starter = self.image_viewer.starting_band_name
        # will not display band selector if band was forced onto image_viewer
        if self.image_viewer.band_provided: band_names = [starter]
        band_selector = ttk.OptionMenu(self, self.band_var, starter, *band_names, command=self.change_band)
        band_selector.pack(side='left')
        ToolTip(band_selector, 'Select band')

        self.color_var = tk.StringVar()
        cmaps = settings.cmaps
        color_selector = ttk.OptionMenu(self, self.color_var, cmaps[0], *cmaps, command=self.change_band)
        color_selector.pack(side='left')
        ToolTip(color_selector, 'Select matplotlib color map')

        # if there is a current view to jump to
        if self.is_linked() and CosiCorNavBar.current_view:
            # switch view to the link view
            self.push_current()
            for ax in self.canvas.figure.get_axes():
                ax._set_view(CosiCorNavBar.current_view)
            self.canvas.draw_idle()
            self.push_current()
        elif zoom:
            for ax in self.canvas.figure.get_axes():
                # zoom in so the main window is looking at the center of the frame with width/height of 1/5 the full image
                scale = 5
                (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
                width, height = x1 - x0, y1 - y0
                nwidth, nheight = width // scale, height // scale
                x0, y0 = x0 + (width - nwidth) // 2, y0 + (height - nheight) // 2
                x1, y1 = x0 + nwidth, y0 + nheight
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)

    # unmodified from NavigationToolbar2Tk except ttk instead of tk
    def _rescale(self):
        """
        Scale all children of the toolbar to current DPI setting.

        Before this is called, the Tk scaling setting will have been updated to
        match the new DPI. Tk widgets do not update for changes to scaling, but
        all measurements made after the change will match the new scaling. Thus
        this function re-applies all the same sizes in points, which Tk will
        scale correctly to pixels.
        """
        for widget in self.winfo_children():
            if isinstance(widget, (ttk.Button, ttk.Radiobutton)):
                if hasattr(widget, '_image_file'):
                    # Explicit class because ToolbarTk calls _rescale.
                    CosiCorNavBar._set_image_for_button(self, widget)
                else:
                    # Text-only button is handled by the font setting instead.
                    pass
            elif isinstance(widget, ttk.Frame):
                widget.configure(height='22p', pady='1p')
                widget.pack_configure(padx='4p')
            elif isinstance(widget, ttk.Label):
                pass  # Text is handled by the font setting instead.
            else:
                _log.warning('Unknown child class %s', widget.winfo_class)
        self._label_font.configure(size=10)

    # unmodified from NavigationToolbar2Tk except ttk instead of tk
    def _update_buttons_checked(self):
        # sync button checkstates to match active mode
        for text, mode in [('Zoom', _Mode.ZOOM), ('Pan', _Mode.PAN)]:
            if text in self._buttons:
                if self.mode == mode:
                    self._buttons[text].var.set(1)
                else:
                    self._buttons[text].var.set(None)

    # unmodified from NavigationToolbar2Tk
    def pan(self, *args):
        super().pan(*args)
        self._update_buttons_checked()

    # unmodified from NavigationToolbar2Tk
    def zoom(self, *args):
        super().zoom(*args)
        self._update_buttons_checked()

    # unmodified from NavigationToolbar2Tk
    def set_message(self, s):
        self.message.set(s)

    # unmodified from NavigationToolbar2Tk
    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.remove_rubberband()
        height = self.canvas.figure.bbox.height
        y0 = height - y0
        y1 = height - y1
        self.lastrect = self.canvas._tkcanvas.create_rectangle(x0, y0, x1, y1)

    # unmodified from NavigationToolbar2Tk
    def remove_rubberband(self):
        if hasattr(self, "lastrect"):
            self.canvas._tkcanvas.delete(self.lastrect)
            del self.lastrect

    # unmodified from NavigationToolbar2Tk except ttk instead of tk
    def _set_image_for_button(self, button):
        """
        Set the image for a button based on its pixel size.

        The pixel size is determined by the DPI scaling of the window.
        """
        if button._image_file is None:
            return

        # Allow _image_file to be relative to Matplotlib's "images" data
        # directory.
        path_regular = button._image_file
        path_large = f"{button._image_file[:-4]}_large.png"
        size = button.winfo_pixels('18p')
        # Use the high-resolution (48x48 px) icon if it exists and is needed
        with ImageTk.Image.open(path_large if (size > 24 and path_large.exists())
                                else path_regular) as im:
            image = ImageTk.PhotoImage(im.resize((size, size)), master=self)
        # button.configure(image=image, height='18p', width='18p') #TODO add back height
        button.configure(image=image)
        button._ntimage = image  # Prevent garbage collection.

    # unmodified from NavigationToolbar2Tk except ttk instead of tk
    def _Button(self, text, image_file, toggle, command):
        if not toggle:
            b = ttk.Button(master=self, text=text, command=command)
        else:
            # There is a bug in tkinter included in some python 3.6 versions
            # that without this variable, produces a "visual" toggling of
            # other near checkbuttons
            # https://bugs.python.org/issue29402
            # https://bugs.python.org/issue25684
            var = tk.IntVar(master=self)
            b = ttk.Radiobutton(master=self, text=text, command=command, variable=var)  # in tk had indicatoron=False
            b.var = var
        b._image_file = image_file
        if image_file is not None:
            # Explicit class because ToolbarTk calls _Button.
            CosiCorNavBar._set_image_for_button(self, b)
        else:
            b.configure(font=self._label_font)
        b.pack(side=tk.LEFT)
        return b

    # unmodified from NavigationToolbar2Tk except ttk instead of tk
    def _Spacer(self):
        # Buttons are also 18pt high.
        s = ttk.Frame(master=self, height='18p', relief=tk.RIDGE, style='Dark.TFrame')
        s.pack(side=tk.LEFT, padx='3p')
        return s

    # unmodified from NavigationToolbar2Tk
    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes().copy()
        default_filetype = self.canvas.get_default_filetype()

        # Tk doesn't provide a way to choose a default filetype,
        # so we just have to put it first
        default_filetype_name = filetypes.pop(default_filetype)
        sorted_filetypes = ([(default_filetype, default_filetype_name)]
                            + sorted(filetypes.items()))
        tk_filetypes = [(name, '*.%s' % ext) for ext, name in sorted_filetypes]

        # adding a default extension seems to break the
        # asksaveasfilename dialog when you choose various save types
        # from the dropdown.  Passing in the empty string seems to
        # work - JDH!
        # defaultextension = self.canvas.get_default_filetype()
        defaultextension = ''
        initialdir = os.path.expanduser(mpl.rcParams['savefig.directory'])
        initialfile = self.canvas.get_default_filename()
        fname = tk.filedialog.asksaveasfilename(
            master=self.canvas.get_tk_widget().master,
            title='Save the figure',
            filetypes=tk_filetypes,
            defaultextension=defaultextension,
            initialdir=initialdir,
            initialfile=initialfile,
        )

        if fname in ["", ()]:
            return
        # Save dir for next time, unless empty str (i.e., use cwd).
        if initialdir != "":
            mpl.rcParams['savefig.directory'] = (
                os.path.dirname(str(fname)))
        try:
            # This method will handle the delegation to the correct type
            self.canvas.figure.savefig(fname)
        except Exception as e:
            tk.messagebox.showerror("Error saving file", str(e))

    # unmodified from NavigationToolbar2Tk
    def set_history_buttons(self):
        state_map = {True: tk.NORMAL, False: tk.DISABLED}
        can_back = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack._elements) - 1

        if "Back" in self._buttons:
            self._buttons['Back']['state'] = state_map[can_back]

        if "Forward" in self._buttons:
            self._buttons['Forward']['state'] = state_map[can_forward]

    # checks if the parent image_hub is linked
    def is_linked(self):
        assert hasattr(self.image_viewer.parent,
                       'is_linked'), "image viewer's parent Window must have a BooleanVar or SimpleContainer with boolean controlling linking"
        return self.image_viewer.parent.is_linked.get()

    # gets all toolbars of parent image_hub's children if linked, otherwise self
    def get_toolbars(self):
        if self.is_linked():
            image_viewers = list(filter(lambda x: isinstance(x, ImageViewer), self.image_viewer.parent.children))
        else:
            image_viewers = [self.image_viewer]
        return list(map(lambda x: x.toolbar, image_viewers))

    # Adapted from super().press_zoom to allow subview moving
    def press_zoom(self, event):
        if event.button != MouseButton.MIDDLE:
            super().press_zoom(event)
        else:
            self.press_pan(event)

    # Adapted from super().release_zoom to work on all of the image hub's children when linked
    def release_zoom(self, event):
        if self._zoom_info is None:
            return

        # We don't check the event button here, so that zooms can be cancelled
        # by (pressing and) releasing another mouse button.
        self.canvas.mpl_disconnect(self._zoom_info.cid)
        self.remove_rubberband()

        x0, y0 = self._zoom_info.start_xy
        key = event.key
        # Force the key on colorbars to ignore the zoom-cancel on the
        # short-axis side
        if self._zoom_info.cbar == "horizontal":
            key = "x"
        elif self._zoom_info.cbar == "vertical":
            key = "y"
        # Ignore single clicks: 5 pixels is a threshold that allows the user to
        # "cancel" a zoom action by zooming by less than 5 pixels.
        if ((abs(event.x - x0) < 5 and key != "y") or
                (abs(event.y - y0) < 5 and key != "x")):
            self.canvas.draw_idle()
            self._zoom_info = None
            return

        # direction = self._zoom_info.direction

        # Convert bounds to data coords.
        transformer = self._zoom_info.axes[0].transData.inverted()
        (x0, y0), (x1, y1) = transformer.transform([(x0, y0), (event.x, event.y)])
        (x0, x1), (y0, y1) = tuple(sorted((x0, x1))), tuple(sorted((y0, y1)))

        view = (x0, x1, y0, y1)
        CosiCorNavBar.current_view = view
        prev_width = self.canvas._tkcanvas.winfo_width()
        prev_height = self.canvas._tkcanvas.winfo_height()

        for toolbar in self.get_toolbars():  # TODO: implement direction out
            toolbar.canvas.prev_width, toolbar.canvas.prev_height = prev_width, prev_height
            toolbar.set_linked_view()

            toolbar.canvas.draw_idle()
            toolbar._zoom_info = None
            toolbar.push_current()

    def set_linked_view(self):
        """Sets the view of all axes to the current saved view"""
        for ax in self.canvas.figure.get_axes():
            ax._set_view(fit_aspect(CosiCorNavBar.current_view, self.canvas))

        self.register_moved()

    def set_preview_getter(self, f):
        self.my_preview = f

        f().canvas.mpl_connect('resize_event', lambda _: self.register_moved())

    def set_subview_getter(self, f):
        self.my_subview = f

        # f().canvas.mpl_connect('resize_event', lambda _: self.register_moved())
        subview = f()
        subview.canvas.mpl_connect('resize_event', self.subview_resized(subview))

    @splitcall
    def subview_resized(self, subview, event):
        self.register_moved()
        subview.register_moved()

    def register_moved(self, *_):
        if hasattr(self, 'my_subview'):
            subview = self.my_subview()
            if subview and hasattr(self.image_viewer, 'preview_locals'):
                view = self.image_viewer.preview_locals
                view = convert_bounds(True, self.canvas, *view)
                view = fit_aspect(view, subview.canvas)
                subview.ax._set_view(view)
                subview.canvas.draw_idle()

        if hasattr(self, 'my_preview'):
            parent = self.my_preview()
            if parent:
                axis = self.canvas.figure.get_axes()[0]
                x0, x1 = axis.get_xlim()
                y0, y1 = axis.get_ylim()

                self.draw_bounds(parent, x0, y0, x1, y1)

    def draw_bounds(self, parent, x0, y0, x1, y1):
        """Takes in a canvas's parent and global coords for the bounds and draws a preview box of the current view"""
        self.remove_bounds(parent)

        x0, x1, y0, y1 = convert_bounds(False, parent.canvas, x0, x1, y0, y1)
        parent.preview_locals = (x0, x1, y0, y1)

        height = parent.canvas.figure.bbox.height
        y1, y0 = height - y0, height - y1
        parent.prevrect = parent.canvas._tkcanvas.create_rectangle(x0, y0, x1, y1, outline='red')

    def remove_bounds(self, parent):
        if hasattr(parent, "prevrect"):
            parent.canvas._tkcanvas.delete(parent.prevrect)
            del parent.prevrect

    def run_linked(self, set_view, push, on_self=False):
        """Sets the view of all linked toolbars and pushes if the predicate is met.

        Args:
            set_view (bool): Whether or not the view should be set.
            push (bool): Whether or not the current view should be pushed to the undo stack.
            on_self (bool, optional): If False, will not move self and will set current_view from self. Defaults to False.
        """
        if not on_self: CosiCorNavBar.current_view = self.canvas.figure.get_axes()[0]._get_view()
        for bar in self.get_toolbars():
            if not on_self and bar == self: continue
            if bar._nav_stack() is None:
                bar.push_current()
            if set_view:
                bar.set_linked_view()
            if push: bar.push_current()
            bar.canvas.draw_idle()

    # Adapted from super().press_pan to work on all of the image hub's children when linked
    def press_pan(self, event):
        """Callback for mouse button press in pan/zoom mode."""
        if event.button != MouseButton.MIDDLE:
            super().press_pan(event)
            self.register_moved()
            self.run_linked(True, False)
        elif hasattr(self.image_viewer, 'subview'):
            subview = self.image_viewer.subview
            if subview:
                self.id_drag = self.canvas.mpl_connect("motion_notify_event", self.drag_subpan)
                self.id_release = self.canvas.mpl_connect('button_release_event', self.release_subpan)
                self.panning = True

                locals = self.image_viewer.preview_locals
                self.offset, bounds = get_bounds(locals, event.x, event.y)
                view = convert_bounds(True, self.canvas, *bounds)
                subview.run_linked(view, on_self=True)

    # Adapted from super().drag_pan to work on all of the image hub's children when linked
    def drag_pan(self, event):
        """Callback for dragging in pan/zoom mode."""
        super().drag_pan(event)
        self.register_moved()
        self.run_linked(True, False)

    def drag_subpan(self, event):
        if not hasattr(self.image_viewer, 'subview'): return
        subview = self.image_viewer.subview
        if not subview: return

        locals = self.image_viewer.preview_locals
        self.offset, bounds = get_bounds(locals, event.x, event.y, offset=self.offset)
        view = convert_bounds(True, self.canvas, *bounds)
        subview.run_linked(view, on_self=True)

    # Adapted from super().release_pan to work on all of the image hub's children when linked
    def release_pan(self, event):
        """Callback for mouse button release in pan/zoom mode."""
        super().release_pan(event)
        self.register_moved()
        self.run_linked(False, True)

    def release_subpan(self, event):
        self.panning = False
        self.canvas.mpl_disconnect(self.id_drag)
        self.canvas.mpl_disconnect(self.id_release)

    # Adapted from super().home to work on all of the image hub's children when linked
    def home(self, *args):
        """
        Restore the original view.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        super().home(*args)
        self.register_moved()

        self.run_linked(True, True)

    # Adapted from super().back to work on all of the image hub's children when linked
    def back(self, *args):
        """
        Move back up the view lim stack.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        super().back(*args)
        self.register_moved()

        self.run_linked(True, True)

    # Adapted from super().forward to work on all of the image hub's children when linked
    def forward(self, *args):
        """
        Move forward in the view lim stack.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        super().forward(*args)
        self.register_moved()

    def change_band(self, e):
        def clear(axes):
            for axis in axes: axis.cla()

        # find all relevant windows
        windows = [self.image_viewer]
        subview = preview = None
        if hasattr(self, 'my_subview'):
            subview = self.my_subview()
        if hasattr(self, 'my_preview'):
            preview = self.my_preview()
        if subview: windows.append(subview)
        if preview: windows.append(preview)

        # clear all axes
        axes = [window.ax for window in windows]
        views = [ax._get_view() for ax in axes]
        clear(axes)

        # draw new bands
        new_band = self.image_viewer.bands[self.band_var.get()]
        color = self.color_var.get()
        self.image_viewer.draw(new_band, axes, color)

        # reset views and draw
        for ax, view in zip(axes, views):
            ax.set(title="", xticks=[], yticks=[])
            ax._set_view(view)
        for window in windows: window.canvas.draw_idle()
