import json
import threading

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as tkfile

from geoCosiCorr3D.geoImageCorrelation.geoCorr_utils import get_bands, pow2, trycatch
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.tk_utils import get_entries, project_path, SimpleContainer, \
    TimedToolTip, findInFrame, removeInvalid, splitcall, open_local_image, addInvalid, reset_option_menu, Window
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.cc_viewer import ImageHub


def param_to_config(paramToConfig):
    """Converts values in entries into a configuration file.

    Args:
        paramToConfig (dict(string, string)): takes in a dictionary pairing from entry names to config names. An '&' preceding an entry name means it is a bool represented in IntVar form.
    """
    entries = readAll()

    config = {}
    for param, fullPath in paramToConfig.items():
        isBool = param[0] == '&'
        for param in param.split(','):
            path = fullPath.split('.')
            current = config
            for dir in path[:-1]:
                current.setdefault(dir, {})
                current = current[dir]
            path = path[-1]
            name = param[1:] if isBool else param
            if name not in entries: continue
            value = entries[name]
            if isBool: value = bool(value)
            if value == '': value = None
            if path in current:
                currentValue = current[path]
                if not isinstance(currentValue, list):
                    currentValue = [currentValue]
                currentValue.append(value)
                value = currentValue
            current[path] = value
    return config


def load_defaults(name, paramToConfig):
    """load the default values into entries to auto-fill parameters.

    Args:
        name (str): File name holding defaults.
        paramToConfig (dict(str, str)): The parameter name to configuration parameter name map.
    """
    entries = get_entries()
    entries.clear()
    with open(project_path(name), "r") as f:
        load_config(json.load(f), paramToConfig, help_strings=True)
    return entries


def load_config(config, paramToConfig, help_strings=False):
    """Loads a configuration dictionary into entries.

    Args:
        config (dict(str, str)): The configuration dictionary.
        paramToConfig (dict(str, str)): The parameter name to configuration parameter name map.
        help_strings (bool): Used internally when the config is a help config that contains defaults and help strings.
    """
    entries = get_entries()
    for param, fullPath in paramToConfig.items():
        path = fullPath.split('.')
        current = config
        for dir in path:
            current = current[dir]

        if help_strings: current, help = current

        # pairs of entry name to value
        pairs = [(param, current)]
        if isinstance(current, list):
            pairs = zip(param.split(','), current)

        for name, value in pairs:
            if name[0] == '&':  # if is bool
                name = name[1:]
                value = int(value)

            entries[name] = SimpleContainer(value)
            if help_strings: entries[f"{name}.help_string"] = SimpleContainer(help)


def readAll():
    """Pulls all set values as a dictionary from name to value."""
    entries = get_entries()
    return {name: entry.get() for name, entry in entries.items()}


def read(name):
    """Pulls one value from the parameter name."""
    entries = get_entries()
    return entries[name].get()


def create_rows(frame, rows, entry_prefix=None):
    """Generates rows for a frame. See `createRow` below.

    Args:
        frame (tk.Frame): The frame to generate on.
        rows (tk.Frame): List of rows to generate. Each row is a tuple with (prompt, params).
        entry_prefix (str, Optional): Extra prefix for these parameter in entries.
    """
    for prompt, params in rows:
        create_row(frame, prompt, params, entry_prefix=entry_prefix)


def create_row(frame, prompt, params, entry_prefix=None):
    """Generates a row on a gridded frame. 
    createRow should be called AFTER validation and invalidation commands are setup if the frame starts in an invalid state.

    Args:
        frame (tk.Frame): The frame to generate on.
        prompt (str): Prompt string to write before all params or None.
        params (list(tuple(str, args))): List of pairs of parameter name to arguments. Each argument is a rowGenerator command. See the bottom of this page for more info.
        entry_prefix (str, Optional): Extra prefix for this parameter in entries.
    """
    global rows, pixelVirtual
    entries = get_entries()
    if 'pixelVirtual' not in globals(): pixelVirtual = tk.PhotoImage(width=1, height=1)
    rows.setdefault(frame, 0)
    row = rows[frame]
    if not hasattr(frame, 'on_validated'): frame.on_validated = lambda: None
    if not hasattr(frame, 'on_invalidated'): frame.on_invalidated = lambda: None
    prefix = ""
    col = 0
    for name, options in params:
        var, item, _, prefix, prompt, col = create_item(frame, name, options, row, col, prefix=prefix, prompt=prompt)

        if var == None: continue

        name = f"{prefix}{name}"
        help_name = f"{name}.help_string"
        if entry_prefix == None:
            full_name = name
        else:
            full_name = f'{entry_prefix}.{name}'

        # load default or previously set value
        if full_name in entries:
            var.set(entries[full_name].get())
        elif name in entries:
            var.set(entries[name].get())
        # create help tooltip
        if help_name in entries:
            items = [item]
            while items:
                item = items[0]
                if isinstance(item, tk.Widget):
                    if item.children:
                        items.extend(list(item.children.values()))
                    elif not hasattr(item, 'info_tip'):
                        item.info_tip = TimedToolTip(item, entries[help_name].get())
                # cycle text for validation
                if isinstance(item, tk.Entry):
                    item.insert(0, '-')
                    item.delete(0, 1)
                items = items[1:]

        # set entry to new var
        entries[full_name] = var

    frame.rowconfigure(rows[frame], weight=1)
    rows[frame] += 1


# holds the row generation counters
rows = {}


def create_item(frame, name, options, row, col, prefix="", prompt=None):
    """Creates a row item given the specified command

    Args:
        frame (Frame): The frame on which to grid the item.
        name (str): The label that should be given to the row item.
        options (str?): The command/options.
        row (int): The row to grid it on.
        col (int): the column to grid it on.
        prefix (str, optional): The prefix to add to the name. Defaults to "".
        prompt (str, optional): The prompt string. Defaults to None.

    Returns:
        tuple(Var/SimpleContainer, Widget, Label, str, str, int): The variable, the item, the label, the new prefix, new prompt, and new column.
    """
    label = None
    txt = f"{name}:"
    if prompt:
        txt = f"{prompt}   {txt}"
        if not name: txt = f"{prompt}:"
        prefix = f"{prompt}."
        prompt = None

    if callable(options):
        options = ('b', name, options)
        col += 1
    elif txt != ":":
        label = ttk.Label(frame, text=txt)
        label.grid(row=row, column=col, sticky="e", pady=3, padx=(5, 1))
        col += 1

    if isinstance(options, dict):
        cmd = ' '
    else:
        cmd = options[0]
        options = options[1:]

    var, item = rowgenerators[cmd](cmd, options, frame)

    item.grid(row=row, column=col, sticky="news", padx=2)
    frame.columnconfigure(col, weight=1)

    col += 1

    return var, item, label, prefix, prompt, col


# map of generator command character to row generator
rowgenerators = {}


def rowGenerator(char):
    """Annotation that registers this function as a rowGenerator with the specified command character.

    Args:
        char (str): The command character that references this function.
    """

    # this level is the annotation and f is the defined command
    def annotation(f):
        rowgenerators[char] = f
        return f

    return annotation


# map of validator command character to validator
validators = {}


# register the command character for a validator (used in createRow) and add validity effects
def validator(char, message):
    """Annotation that registers the command character for a validator (used in createRow) and adds validity effects.

    Args:
        char (str): The validator character that references this function.
        message (str): The tooltip message to display on hover if validation fails.
    """

    # this level is the annotation and f is the defined validator
    def annotation(f):
        # takes in the parser (probably int() or float()) and the current frame
        def contextInput(parser, frame):
            # parses the input and runs the validator
            def runValidator(value, name):
                value = trycatch(parser, value, Exception)
                success = value != None and f(value)
                entry = findInFrame(frame, name)
                if success:
                    removeInvalid(entry, frame.on_validated)
                else:
                    addInvalid(entry, message, frame.on_invalidated)
                return True

            return runValidator

        # register validator
        validators[char] = contextInput
        return contextInput

    return annotation


def entry_shift():
    """Marks the current state of entries. Returns a function that, 
    when called again, copies all changed entries with a prefix.

    Returns:
        (str) -> (): Shift completion function
    """
    entries = get_entries()
    old = entries.copy()

    def complete(prefix):
        """Complete and shift to a new prefix"""
        to_add = []
        for name, val in entries:
            if name in old and old[name] == val: continue
            to_add.append((name, val))
        for name, val in to_add:
            entries[f'{prefix}.{name}'] = val

    return complete


def threaded(f):
    """Annotation that makes a function run on a separate thread"""

    def g(*params, callback=None, **kargs):
        def new_f():
            ret = f(*params, **kargs)
            if callback: callback()
            return ret

        t1 = threading.Thread(target=new_f)
        t1.start()

    return g


### ROW GENERATOR COMMANDS HERE ###


# 'c0' or 'c1' for a checkbox default off or on
@rowGenerator('c')
def checkboxInput(_, options, frame):
    var = tk.IntVar(value=int(options))
    box = ttk.Checkbutton(frame, variable=var)
    return (var, box)


# 'rOPTION1/OPTION2/...OPTIONn' for radio buttons
@rowGenerator('r')
def radioInput(_, options, frame):
    options = options.split('/')
    var = tk.StringVar(value=options[0])

    f = ttk.Frame(frame)
    for i, option in enumerate(options):
        ttk.Radiobutton(f, text=option, variable=var, value=option).grid(row=0, column=i, sticky='news')
    f.rowconfigure(0, weight=1)

    return (var, f)


# 'dOPTION1/OPTION2/...OPTIONn' for dropdown
@rowGenerator('d')
def dropdownInput(_, options, frame):
    options = options.split('/')
    var = tk.StringVar(value=options[0])

    menu = ttk.OptionMenu(frame, var, options[0], *options)

    return (var, menu)


# 'px[f]y'
# x is 'l' for load, 's' for save, and ' ' for neither
# if f is included, this path represents a folder
# y is a string validator (see below). If the validator is t, adds an imageviewer button
@rowGenerator('p')
def pathInput(_, options, frame):
    option = options[0]
    validator = options[1]
    folder = False
    if options[1] == 'f':
        validator = options[2]
        folder = True

    @splitcall
    def open(f):
        filename = f()  # todo use options for filetype options
        entry.delete(0, tk.END)
        entry.insert(0, filename)
        entry.xview_moveto(1)

    f = ttk.Frame(frame)
    var = tk.StringVar()

    text = None

    if option == 'l':
        text, fn = 'Load', tkfile.askopenfilename
    elif option == 's':
        text, fn = 'Save', tkfile.asksaveasfilename

    if folder: fn = tkfile.askdirectory

    # register the validator
    v = validators.get(validator, validatePass)
    v = v(str, frame)
    vcmd = (frame.register(v), '%P', '%W')

    if text: ttk.Button(f, text=text, command=open(fn)).grid(row=0, column=0, sticky='ew')
    entry = ttk.Entry(f, textvariable=var, validate='all', validatecommand=vcmd, width=30)
    entry.grid(row=0, column=1, sticky='news')
    f.columnconfigure(1, weight=1)

    return (var, f)


# 'PPATH' where PATH is the entries name containing the image path's container
@rowGenerator('P')
def bandInput(_, options, frame):
    # cache eye image and make hub
    if not hasattr(frame, 'eye_image'):
        frame.eye_image = open_local_image("./geoImageCorrelation_GUI/assets/icons/eye.png")
        frame.eye_closed_image = open_local_image("./geoImageCorrelation_GUI/assets/icons/eye_closed.png")
        frame.hub = Window.embed_window(None, ttk.Frame(frame), ImageHub)

    f = ttk.Frame(frame)
    string_var = tk.StringVar()  # band name var
    entries_var = tk.IntVar(value=1)  # band number var

    # create options menu
    menu = ttk.OptionMenu(f, string_var, "---------", [])
    menu.pack(fill='both', expand=1, side='left')

    # create eye button
    view = ttk.Label(f, image=frame.eye_closed_image)
    view.pack(side='left')

    # If path changed, load new bands
    path_var = get_entries()[options]

    def path_changed(*_):
        path = path_var.get()
        bands = get_bands(path)
        if not bool(bands):
            view.configure(image=frame.eye_closed_image)
            return
        view.configure(image=frame.eye_image)
        reset_option_menu(menu, string_var, list(bands.keys()))

    path_var.trace_add('write', path_changed)

    # if band name selection changed, set entries var to band number
    def band_changed(*_):
        path = path_var.get()
        bands = get_bands(path)
        if not bool(bands):
            view.configure(image=frame.eye_closed_image)
            return
        entries_var.set(bands[string_var.get()])
        view.configure(image=frame.eye_image)

    string_var.trace_add('write', band_changed)

    # open image viewer on eye press
    def button_press(_):
        path = path_var.get()
        bands = get_bands(path)
        if not bool(bands): return
        band_name = string_var.get()
        frame.hub.new_viewer(path, band_name=band_name)

    view.bind('<ButtonPress>', button_press)
    view.info_tip = TimedToolTip(view, "Click to preview this image.")

    return (entries_var, f)


# 'xyZ' where x is 'i' or 'f' for int or float, y is a validator command character (see validator section below),
#      and Z is the default value with any number of digits
@rowGenerator('i')
@rowGenerator('f')
def numberInput(cmd, options, frame):
    validator, default = options[0], options[1:]
    # check if int or float variable
    if cmd == 'i':
        constructor = tk.IntVar
        parser = int
    elif cmd == 'f':
        constructor = tk.DoubleVar
        parser = float
    var = constructor(value=parser(default))

    # register the validator
    v = validators.get(validator, validatePass)
    v = v(parser, frame)
    vcmd = (frame.register(v), '%P', '%W')

    entry = ttk.Entry(frame, width=4, textvariable=var, validate='all', validatecommand=vcmd)

    return (var, entry)


# creates a button with the label name specfied and the callback as the parameter e.g. ('Okay', callback)
@rowGenerator('b')
def buttonInput(_, options, frame):
    assert isinstance(options,
                      tuple), "Button generator must be called with the text as name and the callback as the options"
    text, f = options

    button = ttk.Button(frame, text=text, command=f)
    return (None, button)


# default generator that simple creates an entry with paremeters specified by a dictionary as options
@rowGenerator(' ')
def defaultInput(cmd, options, frame):
    assert cmd == None, f"Unable to find command '{cmd}'"
    assert isinstance(options, dict), "Options for commandless row must be a dictionary"

    var = tk.StringVar()

    entry = ttk.Entry(frame, textvariable=var, **options)

    return (var, entry)


### VALIDATOR FUNCTIONS HERE ###

# empty validator that checks for nothing
@validator('x', 'This is a bug, this validator cannot fail!')
def validatePass(val):
    return True


# NUMERICAL VALIDATORS #

# checks for a postive value
@validator('p', 'Value must be positive!')
def validatePositive(val):
    return val > 0


# checks for a non-negative value
@validator('n', 'Value must be non-negative!')
def validatePositive(val):
    return val >= 0


# checks for a positive power of 2
@validator('w', 'Value must be a positive power of 2')
def validateWindow(val):
    return val > 0 and pow2(val)


# checks for a value in [0, 1]
@validator('u', 'Value must be within the range [0, 1]')
def validateUnit(val):
    return 1 >= val >= 0


# STRING VALIDATORS #

# checks for a non-empty string
@validator('s', 'Cannot be empty')
def validateNonEmpty(val):
    return bool(val)


# checks for a path to a raster image
@validator('t', 'Must point to valid raster image')
def validateTif(path):
    return bool(get_bands(path))
