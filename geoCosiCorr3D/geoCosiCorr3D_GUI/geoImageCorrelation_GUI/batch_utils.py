import os
from copy import copy

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg

from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.gui_utils import *
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.tkrioplt import ImageHub
import geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.settings as settings
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.tk_utils import get_entries, TimedToolTip, splitcall, \
    open_local_image, VerticalScrolledFrame
from geoCosiCorr3D.geoCosiCorr3D_GUI.geoImageCorrelation_GUI.dnd_lib import dnd_start
from geoCosiCorr3D.geoImageCorrelation.geoCorr_utils import get_bands, project_path, setdefaultattr, splitcall


class BatchItem:
    """Abstract class that represents an item with drag and drop functionality"""

    def __init__(self, draggable=True):
        if not hasattr(BatchItem, 'drag_img'):
            self.drag_img = open_local_image("./geoImageCorrelation_GUI/assets/icons/drag.png")
            self.close_img = open_local_image("./geoImageCorrelation_GUI/assets/icons/close.png")

        self.draggable = draggable
        self.root = None
        self.entries = {}

    def add_item(self, frame, name, options):
        var, item, label, prefix, _, self.col = create_item(frame, name, options, 0, self.col)

        if var == None: return

        name = f"{prefix}{name}"
        # help_name = f"{name}.help_string"
        # load default or previously set value
        if name in self.entries:
            try:
                var.set(self.entries[name].get())
            except tk.TclError:
                pass
        # #create help tooltip
        items = [item]
        while items:
            item = items[0]
            if isinstance(item, tk.Widget):
                if item.children:
                    items.extend(list(item.children.values()))
                else:
                    item.info_tip = TimedToolTip(item, None)
            # cycle text for validation
            if isinstance(item, tk.Entry):
                item.insert(0, '-')
                item.delete(0, 1)
            items = items[1:]

        # set entry to new var
        self.entries[name] = var
        if label: self.draggables.append(label)

    # Drag and drop features below:

    def attach(self, root):
        if self.root: self.root.set_unfocused(True)
        if root is self.root: return
        old_root = self.root
        if old_root is not None: self.detach()
        if root is None: return

        # move_children(root, old_root)

        self.frame = ttk.Frame(root)
        self.frame.parent = self
        self.frame.pack(fill='both', expand=1)
        self.frame.rowconfigure(0, weight=1)

        self.draggables = []
        self.col = 0

        self.root = root
        self.frame.item = self

        self.draw_on(self.frame)
        self.draggables.append(self.frame)

        for child in self.draggables:
            child.bind("<ButtonPress>", self.press)

    def detach(self):
        if self.root is None:
            return
        self.frame.destroy()
        # self.root.delete(id)
        self.root = self.frame = None

    def press(self, event):
        if not self.draggable: return
        if not dnd_start(self, event): return

        self.root.set_focused(separator=False, permanent=True)

    def double_click(self, event):
        return

    def move(self, event):
        return

    def putback(self):
        return

    def where(self, canvas, event):
        return

    def dnd_end(self, target, event):
        if self.root:
            self.root.set_unfocused(True)


class BatchEntry(BatchItem):
    """A drag and drop item displaying a filled in entry"""

    def __init__(self, path, band_name, band, draggable=True):
        super().__init__(draggable)
        self.path = path
        self.band_name = band_name
        if not band_name: self.band_name = str(band)
        self.band = band
        self.text_height = 3

    def draw_on(self, frame):
        frame.on_validated = lambda: None
        frame.on_invalidated = lambda: None

        if self.drag_img:
            drag = ttk.Label(frame, image=self.drag_img)
            drag.grid(row=0, column=self.col, padx=(15, 5), pady=5, sticky='news')
            drag.info_tip = TimedToolTip(drag,
                                         "Left click and hold to drag.\nRight click and hold to duplicate.\nDouble left click to open image.")
            self.col += 1
            self.draggables.append(drag)

        filename = os.path.basename(self.path)
        path = ttk.Label(frame, text=filename)
        path.delete_id = path.bind('<Configure>', self.size_once(path))
        path.grid(row=0, column=self.col, padx=5, pady=5, sticky='news')
        path.info_tip = TimedToolTip(path, self.path)
        self.frame.columnconfigure(self.col, weight=1)
        self.col += 1

        band_name = ttk.Label(frame, text=self.band_name, anchor='center', width=10)
        band_name.delete_id = band_name.bind('<Configure>', self.size_once(band_name))
        band_name.grid(row=0, column=self.col, padx=5, pady=5, sticky='news')
        self.col += 1

        if self.close_img:
            close = ttk.Label(frame, image=self.close_img)
            close.grid(row=0, column=self.col, padx=(5, 15), sticky='news')
            close.bind('<ButtonPress>', self.root.close)
            close.info_tip = TimedToolTip(close, "Click to delete this entry")
            self.col += 1

        self.draggables.extend([path, band_name])

    @splitcall
    def size_once(self, obj, _):
        obj.unbind('<Configure>', obj.delete_id)
        obj.config(wraplength=obj.winfo_width())
        del obj.delete_id

    def double_click(self, event):
        self.open_image()
        return super().double_click(event)

    def open_image(self):
        self.root.selector.hub.new_viewer(self.path, self.band_name)

    def copy(self):
        return BatchEntry(self.path, self.band_name, self.band, self.draggable)


class BatchLabel(BatchEntry):
    """A drag and drop item displaying a sample entry"""

    def __init__(self, draggable=False):
        super().__init__("Full Path/File Name:", "Band Name:", 0, draggable=draggable)
        self.text_height = 1
        self.drag_img = open_local_image("./geoImageCorrelation_GUI/assets/icons/empty.png")
        self.close_img = self.drag_img


class BatchPrompt(BatchItem):
    """A drag and drop item displaying a filled in entry"""

    def __init__(self, draggable=False):
        super().__init__(draggable=draggable)

    def draw_on(self, frame):
        frame.on_validated = lambda: None
        frame.on_invalidated = lambda: None

        self.add_item(frame, "Load", self.root.load)


def move_children(old, new):
    """Takes in two slots and moves all children from old slot to new slot"""
    for child in list(old.children.values()):
        remove_child(child)
        if hasattr(child, 'item'): child.item.attach(new)


def remove_child(child):
    if not hasattr(child, 'item'):
        child.destroy()
        return
    child.item.detach()


class DnDSlot(ttk.Frame):

    def __init__(self, master, selector, *params, **kargs):
        super().__init__(master, *params, **kargs)
        self.selector = selector

    def dnd_accept(self, source, event):
        if isinstance(self.get_batch_item(), BatchLabel):
            return None
        return self

    def dnd_enter(self, source, event):
        # self.focus_set() # Show highlight border
        if source.root == self:
            self.set_focused(separator=False)
        else:
            self.set_focused()
        # x, y = source.where(self, event)
        # x1, y1, x2, y2 = source.canvas.bbox(source.id)
        # dx, dy = x2-x1, y2-y1
        # self.dndid = self.create_rectangle(x, y, x+dx, y+dy)
        self.dnd_motion(source, event)

    def dnd_motion(self, source, event):
        # x, y = source.where(self, event)
        # x1, y1, x2, y2 = self.bbox(self.dndid)
        # self.move(self.dndid, x-x1, y-y1)
        pass

    def dnd_leave(self, source, event):
        # self.master.focus_set() # Hide highlight border
        self.set_unfocused()
        # self.delete(self.dndid)
        # self.dndid = None

    def dnd_commit(self, source, event):
        self.dnd_leave(source, event)
        # x, y = source.where(self, event)
        # source.attach(self, x, y)
        # source.attach(self)

        # if right click, clone
        if event.num == 3:
            self.selector.insert(self, source.copy())
        else:
            self.selector.move(source.root, self)

    def set_focused(self, separator=True, permanent=False):
        if permanent:
            self.locked = True
        if not separator: self.configure(relief='raised')

        if separator:
            children = [c.item for c in self.children.values() if hasattr(c, 'item')]
            for child in children:
                child.detach()

            sep_f = ttk.Frame(self, height='2')
            tk.Label(sep_f, text=' ').pack(fill='both')
            # ttk.Separator(sep_f).pack(fill='both')
            sep_f.pack(fill='x')

            for child in children:
                child.attach(self)

    def set_unfocused(self, permanent=False):
        setdefaultattr(self, 'locked', False)
        if permanent:
            self.locked = False
        if not self.locked:
            self.configure(relief='flat')

        for child in list(self.children.values()):
            if not hasattr(child, 'item'): child.destroy()

    def get_batch_item(self):
        items = [child.parent for child in self.children.values() if hasattr(child, 'parent')]
        return len(items) > 0 and items[0]

    def load(self):
        self.selector.load(self)

    def close(self, e):
        if e.num != 1: return
        self.selector.close(self)


class BatchSelector(ttk.LabelFrame):
    ONE_TO_ONE = None
    ONE_TO_MANY = None
    MANY_TO_MANY = None

    def __init__(self, master, window, param_entry, get_params, *params, **kargs):
        super().__init__(master, *params, text="Input", **kargs)
        if not BatchSelector.ONE_TO_ONE:
            BatchSelector.ONE_TO_ONE = open_local_image("./geoImageCorrelation_GUI/assets/icons/one_to_one.png")
            BatchSelector.ONE_TO_MANY = open_local_image("./geoImageCorrelation_GUI/assets/icons/one_to_many.png")
            BatchSelector.MANY_TO_MANY = open_local_image("./geoImageCorrelation_GUI/assets/icons/many_to_many.png")
        self.window = window
        self.param_entry, self.get_params = param_entry, get_params

        # Provide an image hub for image previewing (will not be visible because frame isnt packed)
        self.hub = self.window.embed_window(ttk.Frame(master), ImageHub)

        # add a frame onto the canvas
        self.slot_f = VerticalScrolledFrame(self, width=900, height=400)
        self.slot_f.pack(side='top', fill='both', expand=1)

        # list of slots per column
        self.columns = [[], [], []]
        self.set_batch_type(BatchSelector.ONE_TO_ONE)
        self.create_slot(0), self.create_slot(0)
        self.create_slot(1), self.create_slot(1)
        self.create_slot(2), self.create_slot(2)

        BatchLabel().attach(self.columns[0][0])
        BatchLabel().attach(self.columns[2][0])
        BatchPrompt().attach(self.columns[0][1])
        BatchPrompt().attach(self.columns[2][1])

        self.slot_f.inner.columnconfigure(0, weight=1, uniform='slots')
        self.slot_f.inner.columnconfigure(2, weight=1, uniform='slots')

        self.params_f = ttk.Frame(master=self)
        self.params_f.pack(side='left', fill='x', expand=1)
        # Map from color number to (Frame, #of rows with that color)
        self.parameter_frames = {}

        # Create batch selector toolbar
        self.toolbar_f = ttk.LabelFrame(window.root, text='Toolbar')
        self.selected = ttk.Label(self.toolbar_f, text="One to One", anchor='center')
        self.selected.pack(pady=5, fill='x', expand=1)

        var = tk.Variable(value="One to One")
        self.add_button(self.toolbar_f, var, BatchSelector.ONE_TO_ONE, "One to One",
                        "For each row, the left band\nwill be correlated with the right band.\nUses that row's configuration.")
        self.add_button(self.toolbar_f, var, BatchSelector.ONE_TO_MANY, "One to Many",
                        "Each left band will be\ncorrelated with each right band.\nUses the left row's configuration.")
        self.add_button(self.toolbar_f, var, BatchSelector.MANY_TO_MANY, "Many to Many",
                        "Each band will be correlated\nwith each other band, regardless of side.\nRepeats with all visible configurations.")

    def add_button(self, frame, var, image, name, message=""):
        button = ttk.Radiobutton(frame, variable=var, image=image, value=name, command=self.set_cmd(image, var))
        button.info_tip = TimedToolTip(button, f"{name}: {message}")
        button.pack(pady=5)

    @splitcall
    def set_cmd(self, image, var):
        self.selected.configure(text=var.get())
        self.set_batch_type(image)

    def pack(self, *params, **kargs):
        super().pack(*params, **kargs)
        self.toolbar_f.pack(padx=(0, 5), pady=5, fill='both')

    def set_batch_type(self, image):
        """Sets the batch type to one of BatchSelector.ONE_TO_ONE, ONE_TO_MANY, MANY_TO_MANY"""
        if image == BatchSelector.ONE_TO_ONE:
            message = 'The configuration for this row.'
        elif image == BatchSelector.ONE_TO_MANY:
            message = 'The configuration for pairs with\nthe base image in this row.'
        elif image == BatchSelector.MANY_TO_MANY:
            message = 'A configuration to run on all pairs.'
        else:
            raise Exception("batch_type must be one of BatchSelector.ONE_TO_ONE, ONE_TO_MANY, or MANY_TO_MANY")

        self.batch_type = image
        self.message = f'{message}\nLeft click -> forward, Right click -> backward.'

        for label in self.columns[1][1:-1]:
            label.info_tip.destroy()
            label.info_tip = TimedToolTip(label, self.message)
            label.configure(image=image)

    def add_slot(self, col, path, band_name, band):
        list = self.columns[col]
        row = len(list)
        self.slot_f.inner.rowconfigure(row, weight=1)

        old_slot = list[-1]
        new_slot = self.create_slot(col)

        # push the prompts onto the new slot
        move_children(old_slot, new_slot)

        # make a new entry and put it where the prompts used to be
        entry = BatchEntry(path, band_name, band)
        entry.attach(old_slot)

    def create_slot(self, col):
        row = len(self.columns[col])

        slot = DnDSlot(self.slot_f, self, borderwidth=2)
        slot.set_unfocused()
        slot.grid(row=row, column=col, sticky='news')

        # if the other column's length is <= this column's length
        if row > 1 and len(self.columns[2 - col]) <= len(self.columns[col]):
            # we need to add another batch type/parameter slot
            self.create_connector(row - 1)

        self.columns[col].append(slot)
        return slot

    def create_connector(self, row):
        lbl = ttk.Label(self.slot_f, image=self.batch_type)
        self.set_param(lbl, 0)
        lbl.info_tip = TimedToolTip(lbl, self.message)
        lbl.grid(row=row, column=1, sticky='news')
        lbl.bind("<Button-1>", lambda _: self.set_param(lbl, lbl.param_num + 1))
        lbl.bind("<Button-3>", lambda _: self.set_param(lbl, lbl.param_num - 1))

        self.columns[1].insert(-1, lbl)

    def set_param(self, lbl, i):
        """Set param_num for future reference and create/destroy required parameter frames."""
        frames = self.parameter_frames

        self.decrement_param_count(lbl)

        # set param_num
        i = i % len(settings.param_colors)
        lbl.param_num = i
        color = settings.param_colors[i]
        lbl.configure(background=color)

        if i in frames:
            # this param window already exists
            f, count = frames[i]
            frames[i] = f, count + 1
            return

            # create frame and parameter row
        f = ttk.Frame(self.params_f)
        f.pack(side='left', ipadx=2, ipady=2)
        if len(frames) > 0:
            ttk.Separator(f, orient='vertical').pack(side='left', fill='y')
        f = ttk.Frame(f, borderwidth=2)
        f.pack(fill='both', expand=1)
        param = lambda: self.make_window(self.get_params(read(f'{i}.Correlator')), i)
        create_row(f, None, [self.param_entry, ("Params", param)], entry_prefix=i)

        # set background color of each item
        style = ttk.Style()
        for child in list(f.children.values()) + [f]:
            name = f'{i}.{child.winfo_class()}'
            style.configure(name, background=color)
            child.configure(style=name)

        # register frame
        frames[i] = (f, 1)

    def decrement_param_count(self, lbl):
        frames = self.parameter_frames
        if hasattr(lbl, 'param_num'):
            old_i = lbl.param_num
            f, count = frames[old_i]
            frames[old_i] = f, count - 1
            if count <= 1:
                # delete the old frame
                f.master.destroy()
                del frames[old_i]

    def make_window(self, constructor, i):
        self.window.new_window(constructor, entry_prefix=i)

    def find(self, slot):
        for col, list in enumerate(self.columns):
            if slot in list:
                row = list.index(slot)
                break
        else:
            raise Exception("Unable to find specified slot")
        return row, col

    def load(self, slot):
        _, col = self.find(slot)

        paths = tkfile.askopenfilenames()
        for path in paths:
            if not path: continue

            bands = get_bands(path)
            if not bands:
                tkmsg.showerror('Invalid', f'The selected file must be a valid image: {os.path.basename(path)}')
                continue

            for band_name, band in bands.items():
                self.add_slot(col, path, band_name, band)

    def close(self, slot):
        row, col = self.find(slot)

        # remove existing items
        for child in list(slot.children.values()): remove_child(child)

        # shift all further slots down 1
        row += 1
        rows = self.columns[col]
        while row < len(rows):
            old_slot = rows[row]
            move_children(old_slot, slot)
            row += 1
            slot = old_slot

        # destroy the empty slot at the end
        rows[-1].destroy()
        del rows[-1]

        # if the other column's length is <= this column's length
        if len(self.columns[2 - col]) <= len(self.columns[col]):
            # we need to remove an image slot
            lbl = self.columns[1][-2]
            self.decrement_param_count(lbl)
            lbl.destroy()
            del self.columns[1][-2]

    def move(self, old_slot, new_slot):
        """Moves an old slot to before the place of the new slot"""
        old_slot.set_unfocused(True), new_slot.set_unfocused(True)
        row_o, col_o = self.find(old_slot)
        row_n, col_n = self.find(new_slot)

        if row_o == row_n and col_o == col_n: return

        # insert new, then close old
        rows_n = self.columns[col_n]
        i = len(rows_n)
        # create new slot at the end of the target column
        self.create_slot(col_n)
        while i > row_n:
            # shift slots up one until the target spot
            move_children(rows_n[i - 1], rows_n[i])
            i -= 1

        if col_o == col_n and row_o > row_n:
            # old_slot is now holding the wrong value, should be slot below
            old_slot = rows_n[row_o + 1]

        # new_slot is now empty, fill it with old_slot's children
        move_children(old_slot, new_slot)
        # close old
        self.close(old_slot)

    def insert(self, slot, item):
        """Inserts item into the location where slot currently is"""
        slot.set_unfocused(True)
        row, col = self.find(slot)
        rows_n = self.columns[col]
        i = len(rows_n)
        # create new slot at the end of the target column
        self.create_slot(col)
        while i > row:
            # shift slots up one until the target spot
            move_children(rows_n[i - 1], rows_n[i])
            i -= 1

        item.attach(slot)

    def pairs(self):
        """Generator that yields one base/target image pair at a time using the specified batch type.
        Entries will automatically be updated to match the current configuration parameters for each pair.

        Yields:
            tuple(BatchEntry, BatchEntry): The base/target image pair.
        """
        olds = []

        def process_pair():
            i = params.param_num

            setfocus_old(False)
            olds.clear()
            olds.extend([base, target, params, self.parameter_frames[i][0]])
            setfocus_old(True)

            # load entries to param_num
            self.set_entries(i, starting_entries)
            return base.get_batch_item(), target.get_batch_item()

        def setfocus_old(set):
            if set:
                # set highlight
                for obj in olds:
                    if hasattr(obj, 'set_focused'):
                        obj.set_focused(separator=False, permanent=True)
                    else:
                        obj.configure(relief='raised')
            else:
                # remove highlight
                for obj in olds:
                    if hasattr(obj, 'set_unfocused'):
                        obj.set_unfocused(permanent=True)
                    else:
                        obj.configure(relief='flat')

        starting_entries = get_entries().copy()
        base_c, params_c, target_c = self.columns
        lener = lambda l: lambda: len(l) - 1
        b_len, t_len, p_len = lener(base_c), lener(target_c), lener(params_c)

        # If you're confused as to why these look so complicated, I have essentially written
        # the loops more manually to include maximal checks for removed or moved slots.
        # This is neccessary because this is running on a non-main thread and thus we have no
        # assurances that things won't magically break if the user edits things.
        i = 1
        if self.batch_type == BatchSelector.ONE_TO_ONE:
            while i < b_len() and i < t_len():
                base, params, target = base_c[i], params_c[i], target_c[i]
                yield process_pair()
                i += 1

        elif self.batch_type == BatchSelector.ONE_TO_MANY:
            while i < b_len():
                j = 1
                while j < t_len():
                    base, params, target = base_c[i], params_c[i], target_c[j]
                    yield process_pair()
                    j += 1
                i += 1

        elif self.batch_type == BatchSelector.MANY_TO_MANY:
            while i < b_len() + t_len() - 1:
                if i < b_len():
                    base = base_c[i]
                else:
                    base = target_c[i - b_len() + 1]

                j = i + 1
                while j < b_len() + t_len() - 1:
                    if j < b_len():
                        target = base_c[j]
                    else:
                        target = target_c[j - b_len() + 1]

                    seen_nums = []
                    k = 1
                    while k < p_len():
                        params = params_c[k]
                        k += 1
                        if params.param_num in seen_nums: continue
                        seen_nums.append(params.param_num)
                        yield process_pair()
                    j += 1
                i += 1

        else:
            raise Exception("batch_type must be one of BatchSelector.ONE_TO_ONE, ONE_TO_MANY, or MANY_TO_MANY")

        setfocus_old(False)
        self.set_entries(-1, starting_entries)

    def set_entries(self, param_num, starting_entries):
        """Takes in a param_num and loads all entries of that number into the main section.

        Args:
            param_num (int): The param_num.
            starting_entries (dict): Copy fo entries at the start of the batch.
        """
        entries = get_entries()
        entries.clear()
        for name, value in starting_entries.items():
            fullname = f'{param_num}.{name}'
            if fullname in starting_entries:
                entries[name] = starting_entries[fullname]
            else:
                entries[name] = value
