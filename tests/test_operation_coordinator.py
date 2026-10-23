from src.services.operation_coordinator import OperationCoordinator


def test_operations_are_mutually_exclusive_and_wait_for_statistics():
    coordinator = OperationCoordinator(statistics_running=True)
    assert coordinator.can_start("auto") == (False, "statistics")
    coordinator.statistics_running = False
    assert coordinator.can_start("auto") == (True, None)
    coordinator.begin("auto")
    assert coordinator.can_start("conversion") == (False, "auto")
    coordinator.finish("auto")
    assert coordinator.can_start("conversion") == (True, None)


def test_shutdown_waits_for_counter_and_dataset_comparison(monkeypatch):
    from types import SimpleNamespace
    from src.widgets import main_window

    class Thread:
        def __init__(self):
            self.stopped = False
            self.waited = False

        def isRunning(self):
            return True

        def quit(self):
            self.stopped = True

        def wait(self):
            self.waited = True

    monkeypatch.setattr(main_window, "QThread", Thread)
    counter, comparison = Thread(), Thread()
    owner = SimpleNamespace(
        task_manager=SimpleNamespace(tasks=lambda: []), _stats_worker=None, _stats_thread=None,
        _dataset_thread=None, _count_thread=counter, _conversion_thread=None,
        _compare_thread=comparison, _video_tools_thread=None, _auto_thread=None,
        _annotation_thread=None, _save_thread=None,
    )

    main_window.MainWindow._stop_background_tasks_for_exit(owner)

    assert counter.stopped and counter.waited
    assert comparison.stopped and comparison.waited
