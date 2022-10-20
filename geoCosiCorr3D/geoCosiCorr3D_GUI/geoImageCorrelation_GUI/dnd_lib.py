# Drag and Drop library from https://github.com/python/cpython/blob/main/Lib/tkinter/dnd.py

__all__ = ["dnd_start", "DndHandler"]


# The factory function

def dnd_start(source, event):
    h = DndHandler(source, event)
    if h.root is not None:
        return h
    else:
        return None


# The class that does the work

class DndHandler:
    root = None

    def __init__(self, source, event):
        if event.num > 3:
            return
        root = event.widget._root()
        try:
            root.__dnd
            return  # Don't start recursive dnd
        except AttributeError:
            root.__dnd = self
            self.root = root
        self.source = source
        self.target = None
        self.initial_button = button = event.num
        self.initial_widget = widget = event.widget
        # self.release_pattern = "<B%d-ButtonRelease-%d>" % (button, button)
        self.release_pattern = f"<ButtonRelease-{button}>"
        self.save_cursor = widget['cursor'] or ""
        widget.bind(self.release_pattern, self.on_release, add='+')
        widget.bind("<Motion>", self.on_motion, add='+')
        widget.bind(f"<Double-Button-1>", self.on_double)
        widget['cursor'] = "hand2"

    def __del__(self):
        root = self.root
        self.root = None
        if root is not None:
            try:
                del root.__dnd
            except AttributeError:
                pass

    def on_motion(self, event):
        x, y = event.x_root, event.y_root
        target_widget = self.initial_widget.winfo_containing(x, y)
        source = self.source
        new_target = None
        while target_widget is not None:
            try:
                attr = target_widget.dnd_accept
            except AttributeError:
                pass
            else:
                new_target = attr(source, event)
                if new_target is not None:
                    break
            target_widget = target_widget.master
        old_target = self.target
        if old_target is new_target:
            if old_target is not None:
                old_target.dnd_motion(source, event)
        else:
            if old_target is not None:
                self.target = None
                old_target.dnd_leave(source, event)
            if new_target is not None:
                new_target.dnd_enter(source, event)
                self.target = new_target

    def on_double(self, event):
        current = event.widget
        while not hasattr(current, 'parent'):
            if not hasattr(current, 'master'): return
            current = current.master
        current.parent.double_click(event)

    def on_release(self, event):
        self.finish(event, 1)

    def cancel(self, event=None):
        self.finish(event, 0)

    def finish(self, event, commit=0):
        target = self.target
        source = self.source
        widget = self.initial_widget
        root = self.root
        try:
            del root.__dnd
            self.initial_widget.unbind(self.release_pattern)
            self.initial_widget.unbind("<Motion>")
            widget['cursor'] = self.save_cursor
            self.target = self.source = self.initial_widget = self.root = None
            if target is not None:
                if commit:
                    target.dnd_commit(source, event)
                else:
                    target.dnd_leave(source, event)

        finally:
            source.dnd_end(target, event)
