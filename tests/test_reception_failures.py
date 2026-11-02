from datetime import datetime, timedelta
import json
from types import SimpleNamespace
import sys

import pytest
from rich.text import Text

from diag_monitor.app import DiagMonitorApp
from diag_monitor.log_writer import LogWriter
from diag_monitor.ros_bridge import RosBridge
from diag_monitor.store import DiagStore


def message(level):
    return SimpleNamespace(status=[SimpleNamespace(
        name="sensor", level=level, message="State", hardware_id="sensor", values=[])])


@pytest.mark.parametrize("operation", ["write", "flush"])
def test_log_failure_keeps_receiving_diagnostics(tmp_path, monkeypatch, operation):
    writer = LogWriter(tmp_path / "diagnostics.jsonl")

    def fail(*args):
        raise OSError("No space left on device")

    monkeypatch.setattr(writer._file, operation, fail)
    store = DiagStore()
    bridge = RosBridge(store, "/diagnostics", writer)
    bridge._on_diagnostics(message(0))
    bridge._on_diagnostics(message(2))
    assert store.snapshot()[0]["sensor"].level == 2
    assert "No space" in bridge.logging_error
    writer.close()


def test_open_failure_is_reported_without_stopping_reception(monkeypatch, tmp_path):
    def fail(*args):
        raise PermissionError("Permission denied")
    monkeypatch.setattr("diag_monitor.log_writer.open", fail, raising=False)
    writer = LogWriter(tmp_path / "diagnostics.jsonl")
    bridge = RosBridge(DiagStore(), "/diagnostics", writer)
    bridge._on_diagnostics(message(2))
    assert "Permission denied" in bridge.logging_error
    writer.close()


def test_failure_does_not_drop_later_array_items_or_retry_output(tmp_path, monkeypatch):
    writer = LogWriter(tmp_path / "diagnostics.jsonl")
    writes = []
    def fail(*args):
        writes.append(args)
        raise OSError("Disk full")
    monkeypatch.setattr(writer._file, "write", fail)
    store = DiagStore()
    bridge = RosBridge(store, "/diagnostics", writer)
    msg = message(0)
    other = message(2).status[0]
    other.name = "other"
    msg.status.append(other)
    bridge._on_diagnostics(msg)
    assert len(store.snapshot()[0]) == 2
    assert store.snapshot()[0]["other"].level == 2
    assert len(writes) == 1
    writer.close()


def test_close_failure_does_not_raise(tmp_path, monkeypatch):
    writer = LogWriter(tmp_path / "diagnostics.jsonl")
    def fail():
        raise OSError("Output unavailable")
    monkeypatch.setattr(writer._file, "close", fail)
    writer.close()
    assert writer.error == "Output unavailable"


@pytest.mark.parametrize("raises", [True, False])
def test_receiver_failure_is_exposed_to_ui(monkeypatch, raises):
    def spin(node):
        if raises:
            raise RuntimeError("Receive failure")
    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(spin=spin))
    bridge = RosBridge(DiagStore(), "/diagnostics")
    bridge._spin()
    assert bridge.receiver_error is not None


def test_expected_shutdown_does_not_report_receiver_failure(monkeypatch):
    def spin(node):
        raise RuntimeError("Context shutting down")
    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(spin=spin))
    bridge = RosBridge(DiagStore(), "/diagnostics")
    bridge._stopping = True
    bridge._spin()
    assert bridge.receiver_error is None


@pytest.mark.parametrize("failure, text, color, level", [
    ("logging_error", "LOG FAILED", "level-warn", 0),
    ("logging_error", "LOG FAILED", "level-error", 2),
    ("receiver_error", "RX STOPPED", "level-error", 0),
])
async def test_failures_remain_visible_instead_of_green(failure, text, color, level):
    store = DiagStore()
    store.update("sensor", level, "State", "sensor", [])
    bridge = SimpleNamespace(topic="/diagnostics", **{failure: "Unavailable [/]"})
    app = DiagMonitorApp(store, ros_bridge=bridge)
    async with app.run_test() as pilot:
        for _ in range(2):
            app._refresh_ui()
            await pilot.pause()
            header = app.query_one("#header")
            assert text in str(header.visual)
            assert header.has_class(color)


@pytest.mark.parametrize("logging_error, receiver_error, markers", [
    ("[Errno 28] No space left on device", None, ["LOG FAILED"]),
    (None, "RuntimeError: Receive failure", ["RX STOPPED"]),
    ("[Errno 28] No space left on device", "RuntimeError: Receive failure",
     ["LOG FAILED", "RX STOPPED"]),
])
async def test_failure_markers_preserve_rendered_counts_at_80_columns(
    logging_error, receiver_error, markers,
):
    store = DiagStore()
    store.update("warn", 1, "State", "hw", [])
    store.update("error", 2, "State", "hw", [])
    bridge = SimpleNamespace(topic="/diagnostics_agg", logging_error=logging_error,
                             receiver_error=receiver_error)
    app = DiagMonitorApp(store, ros_bridge=bridge)
    async with app.run_test(size=(80, 24)) as pilot:
        app._refresh_ui()
        await pilot.pause()
        header = app.query_one("#header")
        rendered = "".join(segment.text for segment in header.render_lines(
            header.region.reset_offset)[0])
        for expected in ("/diagnostics_agg", "2 items", "⚠ 1", "✗ 1", *markers):
            assert expected in rendered
        assert header.has_class("level-error")


def test_unknown_levels_are_logged_and_later_array_items_are_received(tmp_path):
    path = tmp_path / "diagnostics.jsonl"
    writer = LogWriter(path)
    store = DiagStore()
    bridge = RosBridge(store, "/diagnostics", writer)
    first = message(4)
    later = message(2).status[0]
    later.name = "other"
    first.status.append(later)
    try:
        bridge._on_diagnostics(first)
        bridge._on_diagnostics(message(2))
        assert store.snapshot()[0]["other"].level == 2
    finally:
        writer.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["new"] == "4"
    assert records[1]["name"] == "other"
    assert records[2]["prev"] == "4"
    assert records[2]["new"] == "ERROR"


@pytest.mark.parametrize("failure, title", [
    ("logging_error", "Logging failed"),
    ("receiver_error", "Receiver stopped"),
])
async def test_failure_details_notify_once_per_appearance_or_change(
    monkeypatch, failure, title,
):
    bridge = SimpleNamespace(topic="/diagnostics", logging_error=None, receiver_error=None)
    app = DiagMonitorApp(DiagStore(), ros_bridge=bridge)
    notifications = []
    monkeypatch.setattr(app, "notify", lambda message, **kwargs: notifications.append(message))
    async with app.run_test() as pilot:
        for error in ("Unavailable [/]", "Changed failure", None, "Unavailable [/]"):
            setattr(bridge, failure, error)
            for _ in range(2):
                app._refresh_ui()
                await pilot.pause()
        assert [Text.from_markup(message).plain for message in notifications] == [
            f"{title}: Unavailable [/]",
            f"{title}: Changed failure",
            f"{title}: Unavailable [/]",
        ]


@pytest.mark.parametrize("failure, title", [
    ("logging_error", "Logging failed"),
    ("receiver_error", "Receiver stopped"),
])
async def test_failure_details_render_literally_in_real_toast(failure, title):
    error = "[Errno 28] No space [Hz] [/] C:\\temp\\ \\[path] \\"
    bridge = SimpleNamespace(topic="/diagnostics", **{failure: error})
    app = DiagMonitorApp(DiagStore(), ros_bridge=bridge)
    async with app.run_test(notifications=True) as pilot:
        app._refresh_ui()
        await pilot.pause()
        toast = app.query_one("Toast")
        assert toast.render().plain == f"{title}: {error}"


async def test_failure_details_support_legacy_notify_without_markup_option(monkeypatch):
    error = "[Errno 28] No space [Hz] [/] C:\\temp\\ \\[path] \\"
    bridge = SimpleNamespace(topic="/diagnostics", logging_error=error)
    app = DiagMonitorApp(DiagStore(), ros_bridge=bridge)
    rendered = []

    def legacy_notify(message, *, severity, timeout):
        rendered.append(Text.from_markup(message).plain)

    monkeypatch.setattr(app, "notify", legacy_notify)
    async with app.run_test() as pilot:
        app._refresh_ui()
        await pilot.pause()
        assert rendered == [f"Logging failed: {error}"]


@pytest.mark.parametrize("receiver_error", [None, "RuntimeError: Receive failure"])
@pytest.mark.parametrize("logging_error", [None, "No space left on device"])
async def test_no_data_hint_is_only_suppressed_for_stopped_receiver(
    receiver_error, logging_error,
):
    bridge = SimpleNamespace(topic="/diagnostics_agg", receiver_error=receiver_error,
                             logging_error=logging_error)
    app = DiagMonitorApp(DiagStore(), ros_bridge=bridge)
    async with app.run_test() as pilot:
        app._monitor_start = datetime.now() - timedelta(seconds=30)
        app._refresh_ui()
        await pilot.pause()
        hints = [notification.message for notification in app._notifications
                 if "Press" in notification.message]
        assert bool(hints) is (receiver_error is None)
        if receiver_error:
            assert app.query_one("#header").has_class("level-error")
