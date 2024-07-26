from typing import Iterator

from enum import Enum
import numpy as np


class MetricType(Enum):
    ACCURACY = 1
    LOSS = 2
    LOG = 3


class MetricStore:
    def __init__(self, name: str, metric_type: MetricType):
        self._name = name
        self.type = metric_type
        self._store = []
        assert self.type in MetricType, f"Invalid metric type {self.type}"

    def append(self, metric: float | list) -> None:
        self._store.append(metric)

    def get_value(self) -> None | float | list:
        if len(self._store) == 0:
            return None
        if self.type == MetricType.ACCURACY:
            return np.mean(self._store) * 100
        else:
            return np.mean(self._store)

    def get_name(self) -> str:
        return self._name

    def __str__(self) -> str:
        if self.get_value() is None:
            return f"{self._name}: None"
        if self.type == MetricType.ACCURACY:
            return f"{self._name}: {float(self.get_value()):.2f}%"
        return f"{self._name}: {self.get_value():.4f}"

    def __len__(self) -> int:
        return len(self._store)
    
    def __repr__(self) -> str:
        return self.__str__()


class PerTokenMetricStore(MetricStore):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, metric_type=MetricType.LOG)
        if "precision" in kwargs:
            np.set_printoptions(precision=kwargs["precision"])
        else:
            np.set_printoptions(precision=3)

    def get_value(self) -> None | float:
        if len(self._store) == 0:
            return None
        return np.mean(self._store, axis=0)

    def __str__(self) -> str:
        return f"{self._name}: {self.get_value()}"


class MetricStoreCollection:
    def __init__(self, list_of_metric_stores: list[MetricStore]):
        self.metrics = list_of_metric_stores

    def update(self, metrics: dict[str, float | list]) -> None:
        for k, v in metrics.items():
            key_found = False
            for metric in self.metrics:
                if metric.get_name() == k:
                    metric.append(v)
                    key_found = True
                    break
            assert key_found, f"Key {k} not found in metric stores!"
        lengths = [len(self.metrics[i]) for i in range(len(self.metrics))]
        assert (
            np.unique(lengths).shape[0] == 1
        ), f"All metric stores should have the same length after update!, got lengths: {lengths}"

    def create_metric_store(self, name: str, metric_type: MetricType) -> MetricStore:
        assert [
            len(metric._store) == 0 for metric in self.metrics
        ], "All metric stores should be empty before creating a new one!"
        new_metric = MetricStore(name, metric_type)
        self.metrics.append(new_metric)
        return new_metric

    def __str__(self) -> str:
        return "\n".join([str(metric) for metric in self.metrics])
    
    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict:
        return {metric.get_name(): metric.get_value() for metric in self.metrics}

    def __iter__(self) -> Iterator[MetricStore]:
        return iter(self.metrics)
