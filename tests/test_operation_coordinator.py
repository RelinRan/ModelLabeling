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
