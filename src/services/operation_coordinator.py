from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OperationCoordinator:
    """Single source of truth for mutually exclusive background operations."""

    dataset_loading: bool = False
    auto_labeling: bool = False
    converting: bool = False
    statistics_running: bool = False
    statistics_complete: bool = False

    @property
    def active_operation(self) -> str | None:
        if self.dataset_loading:
            return "dataset"
        if self.auto_labeling:
            return "auto"
        if self.converting:
            return "conversion"
        return None

    def can_start(self, operation: str) -> tuple[bool, str | None]:
        if self.active_operation is not None:
            return False, self.active_operation
        if operation in {"auto", "conversion"} and self.statistics_running:
            return False, "statistics"
        return True, None

    def begin(self, operation: str) -> None:
        if operation == "dataset":
            self.dataset_loading = True
        elif operation == "auto":
            self.auto_labeling = True
        elif operation == "conversion":
            self.converting = True

    def finish(self, operation: str) -> None:
        if operation == "dataset":
            self.dataset_loading = False
        elif operation == "auto":
            self.auto_labeling = False
        elif operation == "conversion":
            self.converting = False

