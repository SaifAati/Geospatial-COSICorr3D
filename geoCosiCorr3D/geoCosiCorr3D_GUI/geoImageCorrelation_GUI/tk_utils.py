import time

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
from ttkthemes import ThemedTk
from PIL import ImageTk, Image

from geoCosiCorr3D.geoImageCorrelation.geoCorr_utils import splitcall, project_path, clamp
import geoCosiCorr3D.geoImageCorrelation.geoCorr_utils as utils


def run(window_constructor):
    """The entrypoint for this library. Creates a tk window and executes the mainloop.

    Args:
        constructor (Window): The window constructor for the main window of the program.
    """
    root = get_root()

    window_constructor(None, root)

    root.mainloop()


def get_root():
    """Initialize tk and styles.

    Returns:
        ThemedTk: the root of the application.
    """
    root = ThemedTk(theme='yaru')
    style = ttk.Style(root)

    # text box with red text
    style.element_create("plain.field", 'from', 'yaru', 'Entry.field')
    style.layout("Red.TEntry",
                 [('Entry.plain.field', {'children': [(
                     'Entry.background', {'children': [(
                         'Entry.padding', {'children': [(
                             'Entry.textarea', {'sticky': 'nswe'})],
                             'sticky': 'nswe'})], 'sticky': 'nswe'})],
                     'border': '2', 'sticky': 'nswe'})])
    style.configure("Red.TEntry", foreground='red', fieldbackground='#f0dfdf')

    # progress bar with text over it
    style.layout('text.Horizontal.TProgressbar', [
        ('Horizontal.Progressbar.trough', {'sticky': 'ew', 'children':
            [("Horizontal.Progressbar.pbar", {'side': 'left', 'sticky': 'ns'})]}),
        ('Horizontal.Progressbar.label', {'sticky': 'nswe'})])
    style.configure('text.Horizontal.TProgressbar', anchor='center')  # , foreground='orange')

    style.layout('text.Vertical.TProgressbar', [
        ('Vertical.Progressbar.trough', {'sticky': 'ns', 'children':
            [("Vertical.Progressbar.pbar", {'side': 'top', 'sticky': 'ew'})]}),
        ('Vertical.Progressbar.label', {'sticky': 'nswe'})])
    style.configure('text.Vertical.TProgressbar', anchor='center')  # , foreground='orange')

    # dark colored frame used as a separator
    style.configure('Dark.TFrame', background='DarkGray')

    style.layout("DEBUGFRAME", [('Frame', {})])
    style.configure('DEBUGFRAME', background='red')
    return root


def findInFrame(frame, name):
    """Searches for an object named `name` recursively in `frame`'s children."""
    for _, obj in frame.children.items():
        if str(obj) == name: return obj
        if isinstance(obj, ttk.Frame):
            o = findInFrame(obj, name)
            if o: return o


def reset_option_menu(menu, menu_var, options, index=None):
    """reset the values in the option menu

    if index is given, set the value of the menu to
    the option at the given index
    """
    start_value = menu_var.get()
    menu = menu["menu"]
    menu.delete(0, "end")
    for string in options:
        menu.add_command(label=string, command=lambda value=string: menu_var.set(value))
    if index is not None:
        menu_var.set(options[index])
    elif start_value not in options:
        menu_var.set(options[0])


@splitcall
def disable_button(button):
    """Creates a callback that disables the specified button."""
    if hasattr(button, 'info_tip'):
        button._old_message = button.info_tip.text
        button.info_tip.text = 'Not all parameter requirements\nto press this button have been met.'
    button['state'] = tk.DISABLED


@splitcall
def enable_button(button):
    """Creates a callback that enables the specified button."""
    if hasattr(button, '_old_message'):
        button.info_tip.text = button._old_message
    button['state'] = tk.NORMAL


# _entries holds the output _entries in form {'prompt.paramName': entry}
_entries = {}
get_entries = lambda: _entries

entries_stack = []
invalids_stack = []


def step_in():
    """Saving option selection (aka pressing Okay instead of Cancel) is done as a stack.
    This adds one more layer to the stack."""
    global _entries, invalids
    entries_stack.append(_entries)
    _entries = _entries.copy()
    invalids_stack.append(invalids)
    invalids = {}


def step_out(save):
    """This moves back one layer from the stack.

    Args:
        save (bool): Save changes into _entries if True.
    """
    global _entries, invalids
    current = _entries
    _entries = entries_stack.pop(-1)
    invalids = invalids_stack.pop(-1)
    if not save: return

    for name, entry in current.items():
        if name in _entries:
            _entries[name].set(entry.get())
        else:
            _entries[name] = SimpleContainer(entry.get())


# map of widget to error message tooltip
invalids = {}


def addInvalid(entry, message, invalidated):
    """Adds an entry to the invalids list, and calles the invalidated event if this is the first invalid item."""
    if len(invalids) == 0:
        invalidated()
    if entry not in invalids:
        tip = entry.info_tip
        tip.text = message
        invalids[entry] = None
        invalids[entry] = tip
        entry.configure(style='Red.TEntry')


def removeInvalid(entry, validated):
    """Removes an entry to the invalids list, and calles the validated event if this was the last invalid item."""
    if entry in invalids:
        invalids[entry].reset_msg()
        del invalids[entry]
        if len(invalids) == 0: validated()
        entry.configure(style='TEntry')


class SimpleContainer:
    """A simple container holds a value with a getter or setter. Exists to be used interchangeably with IntVar, StrVar, etc."""

    def __init__(self, value): self.value = value

    def set(self, value): self.value = value

    def get(self): return self.value

    def __str__(self): return f"SimpleContainer: {self.value}"

    __repr__ = __str__


def open_local_image(path):
    """Returns an openened ImageTk relative to the path of the current script."""
    image = Image.open(project_path(path))
    return ImageTk.PhotoImage(image)


# tkinter objects

class ToolTip:
    """
    Tooltip recipe from
    http://www.voidspace.org.uk/python/weblog/arch_d7_2006_07_01.shtml#e387
    """

    def __init__(self, widget, text):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self.text = text

        self.eid = self.widget.bind('<Enter>', self.showtip, add='+')
        self.lid = self.widget.bind('<Leave>', self.hidetip, add='+')

    def showtip(self, _):
        """Display text in tooltip window."""
        if self.tipwindow or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + self.widget.winfo_width()
        y = y + self.widget.winfo_rooty()
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        try:
            # For Mac OS
            tw.tk.call("::tk::unsupported::MacWindowStyle",
                       "style", tw._w,
                       "help", "noActivates")
        except tk.TclError:
            pass
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         relief=tk.SOLID, borderwidth=1)
        label.pack(ipadx=1)

    def hidetip(self, _):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

    def destroy(self):
        self.widget.unbind('<Enter>', self.eid)
        self.widget.unbind('<Leave>', self.lid)
        if self.tipwindow: self.tipwindow.destroy()


class TimedToolTip:
    """
    ToolTip that appears on hover after a certain amount of time.
    Tooltip recipe from
    https://stackoverflow.com/a/36221216
    """

    def __init__(self, widget, text='widget info'):
        self.waittime = 500  # miliseconds
        self.wraplength = 999  # pixels
        self.widget = widget
        self.text = text
        self.start_text = text
        self.enterid = self.widget.bind("<Enter>", self.enter, add='+')
        self.leaveid = self.widget.bind("<Leave>", self.leave, add='+')
        self.leavepid = self.widget.bind("<ButtonPress>", self.leave, add='+')
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        if not self.text: return
        line_count = self.text.count('\n') + 1
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 5
        y += self.widget.winfo_rooty() - 15 * line_count - 10
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                         background="#ffffff", relief='solid', borderwidth=1,
                         wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw = None
        if tw:
            tw.destroy()

    def reset_msg(self):
        self.text = self.start_text

    def destroy(self):
        if self.tw: self.tw.destroy()
        self.widget.unbind("<Enter>", self.enterid)
        self.widget.unbind("<Leave>", self.leaveid)
        self.widget.unbind("<ButtonPress>", self.leavepid)


# Modified from https://gist.github.com/novel-yet-trivial/3eddfce704db3082e38c84664fc1fdf8
class VerticalScrolledFrame:
    """
    A vertically scrolled Frame that can be treated like any other Frame
    ie it needs a master and layout and it can be a master.
    :width:, :height:, :bg: are passed to the underlying Canvas
    :bg: and all other keyword arguments are passed to the inner Frame
    note that a widget layed out in this frame will have a self.master 3 layers deep,
    (outer Frame, Canvas, inner Frame) so 
    if you subclass this there is no built in way for the children to access it.
    You need to provide the controller separately.
    """

    def __init__(self, master, **kwargs):
        width = kwargs.pop('width', None)
        height = kwargs.pop('height', None)
        self.outer = ttk.Frame(master, **kwargs)

        self.vsb = ttk.Scrollbar(self.outer, orient=tk.VERTICAL)
        self.vsb.pack(fill=tk.Y, side=tk.RIGHT)
        self.canvas = tk.Canvas(self.outer, highlightthickness=0, width=width, height=height, background='#f5f6f7')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas['yscrollcommand'] = self.vsb.set
        # mouse scroll does not seem to work with just "bind"; You have
        # to use "bind_all". Therefore to use multiple windows you have
        # to bind_all in the current widget
        self.canvas.bind("<Enter>", self._bind_mouse)
        self.canvas.bind("<Leave>", self._unbind_mouse)
        self.vsb['command'] = self.canvas.yview

        self.inner = ttk.Frame(self.canvas)
        # pack the inner Frame into the Canvas with the topleft corner 4 pixels offset
        self.wid = self.canvas.create_window(0, 0, window=self.inner, anchor='nw')
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.inner.bind("<Configure>", self._on_frame_configure)

        self.outer_attr = set(dir(ttk.Widget))

    def __getattr__(self, item):
        if item in self.outer_attr:
            # geometry attributes etc (eg pack, destroy, tkraise) are passed on to self.outer
            return getattr(self.outer, item)
        else:
            # all other attributes (_w, children, etc) are passed to self.inner
            return getattr(self.inner, item)

    def _on_canvas_configure(self, event):
        width = event.width
        self.canvas.itemconfig(self.wid, width=width)

    def _on_frame_configure(self, event=None):
        x1, y1, x2, y2 = self.canvas.bbox("all")
        height = self.canvas.winfo_height()
        self.canvas.config(scrollregion=(0, 0, x2, max(y2, height)))

    def _bind_mouse(self, event=None):
        self.canvas.bind_all("<4>", self._on_mousewheel)
        self.canvas.bind_all("<5>", self._on_mousewheel)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mouse(self, event=None):
        self.canvas.unbind_all("<4>")
        self.canvas.unbind_all("<5>")
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        """Linux uses event.num; Windows / Mac uses event.delta"""
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")

    def __str__(self):
        return str(self.outer)


class Window:
    """Base class of all my windows that allows it to load child windows and contains helper functions for creating forms"""

    def __init__(self, parent, top_level, title, multichild=False):
        """Creates a base frame for a window.

        Args:
            top_level (tk.TopLevel): The toplevel or window to draw onto.
            title (str): The title of the window.
            multichild (bool, optional): If true, this window can have multiple children. If false, this window supports the _entries stack for input parameters. Defaults to False.
        """
        root_frame = ttk.Frame(top_level)
        root_frame.pack(fill='both', expand=True)
        self.root = root_frame
        self.top_level = top_level
        top_level.title(title)
        self.children = None
        self.parent = parent
        if multichild: self.children = []

    def new_window(self, constructor, *params, on_close=None, **kargs):
        """Generates a new window using the specified `Window`-descendant constructor, passing in the specified params and kargs.

        Args:
            constructor (constructor): The type of the specified window.
            on_close (f()): An additional callback to run when the window closes.

        Returns:
            Window: The child.
        """
        child = tk.Toplevel(self.root)

        if isinstance(self.children, list):
            child = constructor(self, child, *params, **kargs)
            self.children.append(child)
        else:
            step_in()
            child = constructor(self, child, *params, **kargs)
            self.top_level.withdraw()
            self.children = child

        child.close_callback = on_close
        child.top_level.protocol("WM_DELETE_WINDOW", self.child_close(child, False))

        return child

    def embed_window(self, master, constructor, *params, **kargs):
        """Embeds a Window onto master with specified parameters.

        Args:
            constructor: The Window constructor.

        Returns:
            Window: The embedded window.
        """
        if not hasattr(master, 'title'): master.title = lambda _: None
        window = constructor(self, master, *params, **kargs)

        return window

    @splitcall
    def child_close(self, child, to_save):
        """Closes the current child window and bring this one back into view."""
        if child.close_callback:
            child.close_callback()
        if isinstance(self.children, list):
            self.children.remove(child)
            child.top_level.destroy()
            return
        step_out(to_save)
        self.top_level.deiconify()
        self.children.top_level.destroy()
        self.children = None

    @splitcall
    def back(self, to_save):
        """Go back, aka close the current window and return to the previous"""
        self.parent.child_close(self.parent.children, to_save)()

    def load_template(self, text):
        """Creates the standard template with self.params_f and self.buttons_f with an ok and cancel button."""
        self.params_f = ttk.LabelFrame(self.root, text=text)
        self.params_f.pack(padx=5, pady=5, fill="both")

        buttons_f = ttk.Frame(self.root)
        buttons_f.pack(fill=tk.X, ipadx=5, ipady=5)

        ok_b = ttk.Button(buttons_f, text="Ok", command=self.back(True))
        ok_b.pack(side=tk.RIGHT, padx=10, ipadx=10)

        self.register_on_invalidate(disable_button(ok_b), self.params_f)
        self.register_on_validate(enable_button(ok_b), self.params_f)

        ttk.Button(buttons_f, text="Cancel", command=self.back(False)).pack(side=tk.RIGHT, ipadx=10)

    def make_frame(self, **kargs):
        """Create and pack a basic frame with some default values set."""
        grid = 'row' in kargs or 'column' in kargs
        kargs.setdefault('master', self.root)
        kargs.setdefault('text', None)
        if grid:
            kargs.setdefault('sticky', 'news')
        else:
            kargs.setdefault('fill', 'both')
        kargs.setdefault('padx', 5)
        kargs.setdefault('pady', 5)
        f = ttk.LabelFrame(kargs['master'], text=kargs['text']) if kargs['text'] else ttk.Frame(kargs['master'])
        del kargs['master']
        del kargs['text']
        if grid:
            f.grid(**kargs)
        else:
            f.pack(**kargs)
        return f

    def make_run_bar(self, command, param_f, run_tip, start_msg, complete_msg, horizontal=True):
        """Create a run bar that runs (on a separate thread) a command.

        Args:
            command ((0 -> ())): The callback to run on a separate thread upon press of the 'Run' button.
            param_f (Frame): The 'Run' button will only be enabled when this frame is validated.
            run_tip (str): The message on the run button's hover.
            start_msg (str): The message to show when run is pressed.
            complete_msg (str): The message to show when command is completed.
        """

        def prog(val):
            val = int(clamp(val, 1, 100))
            run_p['value'] = val
            ttk.Style().configure(style, text=str(val) + ' %')
            # if val == 100:
            #     tkmsg.showinfo('Complete', message=complete_msg)

        def run():
            command(callback=thread_callback)
            ttk.Style().configure(style, text='0 %')
            tkmsg.showinfo('Starting...', start_msg)

        def thread_callback():
            time.sleep(0.1)
            tkmsg.showinfo('Complete', message=complete_msg)

        utils.__mark_progress__ = prog

        if horizontal:
            b_side, p_side = tk.RIGHT, tk.LEFT
            fill = 'x'
            padx = ipadx = 10
            pady = ipady = 0
            style = 'text.Horizontal.TProgressbar'
            orient = 'horizontal'
        else:
            b_side, p_side = tk.BOTTOM, tk.TOP
            fill = 'y'
            padx = ipadx = 0
            pady = ipady = 10
            style = 'text.Vertical.TProgressbar'
            orient = 'vertical'

        self.runbar_f = self.make_frame(expand=1, padx=(0, 5), pady=(0, 5))
        run_b = ttk.Button(self.runbar_f, text="Run", command=run)
        run_b.pack(side=b_side, padx=padx, ipadx=ipadx, pady=pady, ipady=ipady)
        run_b.info_tip = TimedToolTip(run_b, text=run_tip)

        run_p = ttk.Progressbar(self.runbar_f, style=style, orient=orient)
        ttk.Style().configure(style, text='')
        run_p.pack(side=p_side, padx=padx, ipadx=ipadx, pady=pady, ipady=ipady, fill=fill, expand=True)
        run_p['value'] = 0

        self.register_on_invalidate(disable_button(run_b), param_f)
        self.register_on_validate(enable_button(run_b), param_f)

    def register_on_validate(self, function, frame):
        """Adds an additional function to the end of the `validated` event handler"""
        if hasattr(frame, 'on_validated'):
            old = frame.on_invalidated

            def chained():
                old()
                function()
        else:
            chained = function
        frame.on_validated = chained

    def register_on_invalidate(self, function, frame):
        """Adds an additional function to the end of the `invalidated` event handler"""
        if hasattr(frame, 'on_invalidated'):
            old = frame.on_invalidated

            def chained():
                old()
                function()
        else:
            chained = function
        frame.on_invalidated = chained

    def redirect_validation(self, from_frame, to_frame):
        """Redirect validation events from from_frame to to_frame"""
        self.register_on_validate(lambda *params, **kargs: to_frame.on_validated(*params, **kargs), from_frame)
        self.register_on_invalidate(lambda *params, **kargs: to_frame.on_invalidated(*params, **kargs), from_frame)
