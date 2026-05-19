from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode


class ModeOverlay:
    def __init__(self, app):
        self.app = app
        self.text = OnscreenText(
            text="",
            pos=(1.27, 0.93),
            scale=0.045,
            fg=(0, 0, 0, 1),
            bg=(1, 1, 1, 0.8),
            align=TextNode.ARight,
            mayChange=True,
            parent=app.aspect2d,
        )

    def update(self, mode_label, mode_index, total_modes, is_dirty=False):
        content = f"[{mode_index}/{total_modes}] {mode_label}"
        if is_dirty:
            content += "\n● Unsaved changes"
        self.text.setText(content)

    def destroy(self):
        self.text.destroy()


class StatusHud:
    def __init__(self, app):
        self.app = app
        self.title = OnscreenText(text="", pos=(-1.3, 0.94), scale=0.05, align=TextNode.ALeft, mayChange=True, parent=app.aspect2d)
        self.controls = OnscreenText(text="", pos=(-1.3, 0.88), scale=0.038, align=TextNode.ALeft, mayChange=True, parent=app.aspect2d)
        self.status = OnscreenText(text="", pos=(0, -0.94), scale=0.04, align=TextNode.ACenter, mayChange=True, parent=app.aspect2d)

    def update(self, title, controls, status):
        self.title.setText(title)
        self.controls.setText(controls)
        self.status.setText(status)

    def destroy(self):
        self.title.destroy()
        self.controls.destroy()
        self.status.destroy()


class SavePrompt:
    """Simple Y/N/Esc prompt state helper."""

    def __init__(self, app):
        self.app = app
        self.active = False
        self.callback = None
        self.prompt = OnscreenText(
            text="",
            pos=(0, 0),
            scale=0.06,
            fg=(1, 1, 0.8, 1),
            align=TextNode.ACenter,
            mayChange=True,
            parent=app.aspect2d,
        )
        self.prompt.hide()

    def show(self, mode_label, callback):
        self.active = True
        self.callback = callback
        self.prompt.setText(f"Unsaved changes in {mode_label}: Y Save / N Discard / Esc Cancel")
        self.prompt.show()

    def hide(self):
        self.active = False
        self.callback = None
        self.prompt.hide()

    def handle_key(self, key):
        if not self.active or self.callback is None:
            return False
        if key in ("y", "Y"):
            cb = self.callback
            self.hide()
            cb("save")
            return True
        if key in ("n", "N"):
            cb = self.callback
            self.hide()
            cb("discard")
            return True
        if key == "escape":
            cb = self.callback
            self.hide()
            cb("cancel")
            return True
        return False

    def destroy(self):
        self.prompt.destroy()
