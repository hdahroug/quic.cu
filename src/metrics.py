"""Metrics registry, timer, and span tracking for serving performance."""

from __future__ import annotations

import itertools
import json
import time
from typing import NotRequired, TypedDict


class TimerLog(TypedDict):
    name: str
    duration: float


class SpanLog(TypedDict):
    id: int
    name: str
    duration: float
    parent_id: int | None


class TimerJson(TypedDict):
    name: str
    duration: float


class SpanJson(TypedDict):
    name: str
    duration: float
    children_spans: NotRequired[list[SpanJson]]


class MetricsJson(TypedDict):
    timers: list[TimerJson]
    spans: list[SpanJson]


class MetricsRegistry:
    def __init__(self) -> None:
        self._timers: list[TimerLog] = []
        self._spans: list[SpanLog] = []

    def add_timer(self, timer_log: TimerLog) -> None:
        self._timers.append(timer_log)

    def add_span(self, span_log: SpanLog) -> None:
        self._spans.append(span_log)

    def dump_json(self, file_name: str) -> MetricsJson:
        metrics_json: MetricsJson = {"timers": [], "spans": []}

        for timer in self._timers:
            metrics_json["timers"].append(
                {"name": timer["name"], "duration": timer["duration"]}
            )

        for span in self._spans:
            if span["parent_id"]:
                continue
            metrics_json["spans"].append(self._build_span_json(span))

        with open(file_name, "w") as file:
            json.dump(metrics_json, file, indent=4)

        return metrics_json

    def _build_span_json(self, span: SpanLog) -> SpanJson:
        children_spans = [s for s in self._spans if s["parent_id"] == span["id"]]
        span_json: SpanJson = {"name": span["name"], "duration": span["duration"]}
        if children_spans:
            span_json["children_spans"] = []
            for child_span in children_spans:
                span_json["children_spans"].append(self._build_span_json(child_span))
        return span_json


class Timer:
    def __init__(self, name: str, registry: MetricsRegistry | None = None) -> None:
        self.name = name
        self.registry = registry
        self.start_time = 0.0
        self.end_time = 0.0

    def __enter__(self) -> Timer:
        if self.registry:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, traceback) -> None:
        if self.registry:
            self.end_time = time.perf_counter()
            self.registry.add_timer(
                {"name": self.name, "duration": self.end_time - self.start_time}
            )


class Span:
    _stack: list[Span] = []
    _id_generator = itertools.count(1)

    def __init__(self, name: str, registry: MetricsRegistry | None = None) -> None:
        self.id = next(Span._id_generator)
        self.name = name
        self.registry = registry
        self.start_time = 0.0
        self.end_time = 0.0
        self.parent_id: int | None = None

    def __enter__(self) -> Span:
        if not self.registry:
            return self
        if Span._stack:
            self.parent_id = Span._stack[-1].id
        self.start_time = time.perf_counter()
        Span._stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, traceback) -> None:
        if not self.registry:
            return
        self.end_time = time.perf_counter()
        self.registry.add_span(
            {
                "id": self.id,
                "name": self.name,
                "duration": self.end_time - self.start_time,
                "parent_id": self.parent_id,
            }
        )
        Span._stack.pop()
