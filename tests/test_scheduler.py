"""Unit tests verifying request admission and slot release policies for batch schedulers."""

from collections import deque
import pytest
import torch

from sequence import SequenceState
from scheduler import ContinuousAdmissionPolicy, StaticAdmissionPolicy


def test_static_scheduler_admits_when_empty():
    policy = StaticAdmissionPolicy()
    waiting = deque(
        [
            SequenceState(prompt_tokens=torch.tensor([[1, 2]]), request_id=1, max_tokens=10),
            SequenceState(prompt_tokens=torch.tensor([[3, 4]]), request_id=2, max_tokens=10),
        ]
    )
    decision = policy.decide(running=[], waiting=waiting, max_batch_size=2)
    assert len(decision.admit) == 2
    assert len(decision.release) == 0


def test_static_scheduler_blocks_while_any_running():
    policy = StaticAdmissionPolicy()
    seq1 = SequenceState(prompt_tokens=torch.tensor([[1]]), request_id=1, max_tokens=10)
    seq2 = SequenceState(prompt_tokens=torch.tensor([[2]]), request_id=2, max_tokens=10)
    seq1.generated_tokens = [1, 2]

    running = [seq1]
    waiting = deque([seq2])

    decision = policy.decide(running=running, waiting=waiting, max_batch_size=2)
    assert len(decision.admit) == 0
    assert len(decision.release) == 0


def test_continuous_scheduler_admits_into_vacant_slots():
    policy = ContinuousAdmissionPolicy()
    seq1 = SequenceState(prompt_tokens=torch.tensor([[1]]), request_id=1, max_tokens=2)
    seq1.generated_tokens = [10, 20]

    seq2 = SequenceState(prompt_tokens=torch.tensor([[2]]), request_id=2, max_tokens=10)
    seq2.generated_tokens = [30]

    seq3 = SequenceState(prompt_tokens=torch.tensor([[3]]), request_id=3, max_tokens=10)
    seq4 = SequenceState(prompt_tokens=torch.tensor([[4]]), request_id=4, max_tokens=10)

    running = [seq1, seq2]
    waiting = deque([seq3, seq4])

    decision = policy.decide(running=running, waiting=waiting, max_batch_size=2)
    assert len(decision.release) == 1
    assert decision.release[0].request_id == 1
    assert len(decision.admit) == 1
    assert decision.admit[0].request_id == 3
