"""Request batch admission policies for static and continuous serving."""
import itertools
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

from sequence import SequenceState


@dataclass
class AdmissionDecision:
    release: list[SequenceState]
    admit: list[SequenceState]


class BatchAdmissionPolicy(ABC):
    @abstractmethod
    def decide(
        self,
        running: list[SequenceState],
        waiting: deque[SequenceState],
        max_batch_size: int,
    ) -> AdmissionDecision:
        pass


class StaticAdmissionPolicy(BatchAdmissionPolicy):
    def decide(
        self,
        running: list[SequenceState],
        waiting: deque[SequenceState],
        max_batch_size: int,
    ) -> AdmissionDecision:
        num_running = 0
        release: list[SequenceState] = []
        for sequence in running:
            if sequence.is_finished:
                release.append(sequence)
            else:
                num_running += 1
        if num_running != 0:
            return AdmissionDecision([], [])
        num_admitted = min(len(waiting), max_batch_size)
        admit = list(itertools.islice(waiting, num_admitted))
        return AdmissionDecision(release, admit)


class ContinuousAdmissionPolicy(BatchAdmissionPolicy):
    def decide(
        self,
        running: list[SequenceState],
        waiting: deque[SequenceState],
        max_batch_size: int,
    ) -> AdmissionDecision:
        num_running = 0
        release: list[SequenceState] = []
        for sequence in running:
            if sequence.is_finished:
                release.append(sequence)
            else:
                num_running += 1

        num_admitted = min(len(waiting), max_batch_size - num_running)
        admit = list(itertools.islice(waiting, num_admitted))
        return AdmissionDecision(release, admit)
