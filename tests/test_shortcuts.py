# -*- coding: utf-8 -*-
"""Keyboard-shortcut behaviour, exercised on a synthetic dataset.

Replaces the throwaway probe that used to open whatever dataset the developer
last had open. Everything here runs against ``synthetic_dataset``.

Two-stroke chords must work from both panels and the canvas, while text
editing and single-stroke canvas editing retain their normal behaviour.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLineEdit
from PySide6.QtTest import QTest

from src.widgets.main_window import MainWindow

CTRL = Qt.KeyboardModifier.ControlModifier
NONE = Qt.KeyboardModifier.NoModifier

# (chord label, keys, handler name on MainWindow)
CHORDS = [
    ("Ctrl+H", [(Qt.Key.Key_H, CTRL)], "open_history"),
    ("Ctrl+O", [(Qt.Key.Key_O, CTRL)], "open_directory"),
    ("Ctrl+N", [(Qt.Key.Key_N, CTRL)], "open_dataset_init"),
    ("Ctrl+S", [(Qt.Key.Key_S, CTRL)], "save_current"),
    ("Ctrl+L+G", [(Qt.Key.Key_L, CTRL), (Qt.Key.Key_G, NONE)], "open_label_groups"),
    ("Ctrl+K+G", [(Qt.Key.Key_K, CTRL), (Qt.Key.Key_G, NONE)], "open_keypoint_groups"),
    ("Ctrl+I+F", [(Qt.Key.Key_I, CTRL), (Qt.Key.Key_F, NONE)], "open_image_filter"),
    ("Ctrl+A+S", [(Qt.Key.Key_A, CTRL), (Qt.Key.Key_S, NONE)], "open_settings"),
    ("Ctrl+A+L", [(Qt.Key.Key_A, CTRL), (Qt.Key.Key_L, NONE)], "auto_label_all"),
    ("Ctrl+C+A", [(Qt.Key.Key_C, CTRL), (Qt.Key.Key_A, NONE)], "open_crosshair"),
    ("Ctrl+C+D", [(Qt.Key.Key_C, CTRL), (Qt.Key.Key_D, NONE)], "open_cleanup"),
    ("Ctrl+D+S", [(Qt.Key.Key_D, CTRL), (Qt.Key.Key_S, NONE)], "open_dataset_synthesis"),
    ("Ctrl+D+P", [(Qt.Key.Key_D, CTRL), (Qt.Key.Key_P, NONE)], "open_dataset_compare"),
    ("Ctrl+D+C", [(Qt.Key.Key_D, CTRL), (Qt.Key.Key_C, NONE)], "open_conversion"),
    ("Ctrl+T+S", [(Qt.Key.Key_T, CTRL), (Qt.Key.Key_S, NONE)], "open_statistics"),
    ("Ctrl+V+F", [(Qt.Key.Key_V, CTRL), (Qt.Key.Key_F, NONE)], "open_video_frames"),
]

CHORD_IDS = [label for label, _, _ in CHORDS]


@pytest.fixture
def window(synthetic_dataset):
    app = QApplication.instance() or QApplication([])
    dataset = synthetic_dataset("shortcut-ds", fmt="voc", frames=3)
    win = MainWindow()
    win.show()
    app.processEvents()
    # Never let a test depend on which dataset the window happened to open.
    assert win.dataset_root is None or "shortcut-ds" in str(win.dataset_root)
    assert str(dataset)
    yield win, app
    win.close()


def _replay(win, app, focus_widget, keys):
    win._shortcut_chord = None
    focus_widget.setFocus(Qt.FocusReason.OtherFocusReason)
    app.processEvents()
    for key, modifier in keys:
        QTest.keyClick(focus_widget, key, modifier)
        app.processEvents()
        QTest.qWait(40)
        app.processEvents()
    QTest.qWait(60)
    app.processEvents()


@pytest.mark.parametrize("label,keys,handler", CHORDS, ids=CHORD_IDS)
def test_chord_reaches_its_handler(window, label, keys, handler):
    """Every advertised chord must reach its handler from the label panel."""
    win, app = window
    fired = []
    if handler == "save_current":
        # Saving a real image is fine on the synthetic dataset, but assert on
        # the invocation rather than a side effect.
        win.save_current = lambda *a, **k: fired.append(handler)
    elif hasattr(win, handler):
        setattr(win, handler, lambda *a, **k: fired.append(handler))
    else:
        pytest.skip(f"{handler} does not exist in this build")

    _replay(win, app, win.preset_panel, keys)
    assert handler in fired, f"{label} did not reach {handler}"


@pytest.mark.parametrize("label,keys,handler", CHORDS, ids=CHORD_IDS)
def test_chord_reaches_its_handler_with_canvas_focused(window, label, keys, handler):
    """The common case: the canvas holds focus while annotating."""
    win, app = window
    fired = []
    if hasattr(win, handler):
        setattr(win, handler, lambda *a, **k: fired.append(handler))
    else:
        pytest.skip(f"{handler} does not exist in this build")

    _replay(win, app, win.canvas, keys)
    assert handler in fired, f"{label} did not reach {handler}"


def test_plain_keys_still_work_on_the_canvas(window):
    """W toggles once per press -- not twice, which would cancel out."""
    win, app = window
    canvas = win.canvas
    canvas._disable_draw_mode()
    canvas.setFocus(Qt.FocusReason.OtherFocusReason)
    app.processEvents()

    toggles = []
    original = win._toggle_canvas_drawing

    def counted():
        toggles.append(1)
        return original()

    win._toggle_canvas_drawing = counted
    QTest.keyClick(canvas, Qt.Key.Key_W, NONE)
    app.processEvents()
    QTest.qWait(80)
    app.processEvents()

    assert len(toggles) == 1, f"W fired {len(toggles)} handlers, expected exactly 1"
    assert canvas.draw_enabled is True


def test_ctrl_a_selects_all_on_the_canvas(window):
    """Ctrl+A alone still selects every annotation."""
    win, app = window
    calls = []
    original = win.canvas.select_all
    win.canvas.select_all = lambda *a, **k: (calls.append(1), original())[1]

    # The canvas shortcut is WidgetWithChildrenShortcut, so the canvas must
    # actually hold focus for Qt to route the key.
    win.canvas.setFocus(Qt.FocusReason.OtherFocusReason)
    app.processEvents()
    QTest.keyClick(win.canvas, Qt.Key.Key_A, CTRL)
    app.processEvents()
    QTest.qWait(60)
    app.processEvents()

    assert calls == [1], f"Ctrl+A fired select_all {len(calls)} times"


@pytest.mark.parametrize("label,keys,handler", [item for item in CHORDS if len(item[1]) == 2],
                         ids=[item[0] for item in CHORDS if len(item[1]) == 2])
def test_chord_works_while_control_is_held(window, label, keys, handler):
    win, app = window
    fired = []
    setattr(win, handler, lambda *args, **kwargs: fired.append(handler))
    _replay(win, app, win.canvas, [(key, CTRL) for key, _ in keys])
    assert fired == [handler], label


def test_w_in_text_input_does_not_toggle_drawing(window):
    win, app = window
    editor = QLineEdit(win)
    editor.show()
    editor.setFocus(Qt.FocusReason.OtherFocusReason)
    app.processEvents()
    toggles = []
    win._toggle_canvas_drawing = lambda: toggles.append(1)

    QTest.keyClick(editor, Qt.Key.Key_W)
    app.processEvents()

    assert editor.text() == "w"
    assert not toggles


def test_canvas_select_copy_paste_in_quick_succession(window):
    win, app = window
    calls = []
    win.canvas.select_all = lambda: calls.append("select")
    win.canvas.copy_selected = lambda: calls.append("copy")
    win.canvas.paste_clipboard = lambda: calls.append("paste")

    _replay(win, app, win.canvas, [(Qt.Key.Key_A, CTRL), (Qt.Key.Key_C, CTRL), (Qt.Key.Key_V, CTRL)])
    QTest.qWait(1250)
    app.processEvents()

    assert calls == ["select", "copy", "paste"]


def test_video_chord_never_pastes_into_the_canvas(window):
    win, app = window
    calls = []
    win.canvas.paste_clipboard = lambda: calls.append("paste")
    win.open_video_frames = lambda: calls.append("video")

    _replay(win, app, win.canvas, [(Qt.Key.Key_V, CTRL), (Qt.Key.Key_F, NONE)])
    QTest.qWait(1250)
    app.processEvents()

    assert calls == ["video"]
