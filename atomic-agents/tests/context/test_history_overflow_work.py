import random

import pytest

from atomic_agents import BaseIOSchema
from atomic_agents.context import ChatHistory


class Entry(BaseIOSchema):
    """A numbered message used to compare retained history."""

    number: int


class MovementList(list):
    """Count elements removed or shifted by ordinary list mutations."""

    work = 0

    def pop(self, index=-1):
        position = index if index >= 0 else len(self) + index
        self.work += len(self) - position
        return super().pop(index)

    def __delitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            assert step == 1
            self.work += len(self) - start if stop > start else 0
        else:
            position = key if key >= 0 else len(self) + key
            self.work += len(self) - position
        return super().__delitem__(key)


@pytest.mark.parametrize("capacity", [0, 32])
def test_lowering_capacity_does_not_repeatedly_shift_the_remaining_history(capacity):
    history = ChatHistory()
    for number in range(2000):
        history.add_message("user", Entry(number=number))
    tracked = MovementList(history.history)
    history.history = tracked
    history.max_messages = capacity

    history.add_message("assistant", Entry(number=2000))

    assert history.history is tracked
    assert [message.content.number for message in tracked] == list(range(2001 - capacity, 2001))
    assert tracked.work <= 2 * 2001


@pytest.mark.parametrize("capacity", [None, 0, 1, 3, 5, 10])
def test_overflow_keeps_newest_messages_and_list_identity(capacity):
    history = ChatHistory()
    for number in range(5):
        history.initialize_turn()
        history.add_message("user", Entry(number=number))
    old_messages = list(history.history)
    same_list = history.history
    history.max_messages = capacity
    history.initialize_turn()
    turn = history.current_turn_id
    added = Entry(number=5)

    history.add_message("assistant", added)

    expected_count = 6 if capacity is None else min(capacity, 6)
    assert history.history is same_list
    assert history.current_turn_id == turn
    assert history.get_message_count() == expected_count
    assert [message.content.number for message in history.history] == list(range(6 - expected_count, 6))
    if expected_count:
        assert history.history[-1].content is added
        assert history.history[-1].role == "assistant"
        assert history.history[-1].turn_id == turn
        for message in history.history[:-1]:
            assert any(message is original for original in old_messages)
    else:
        assert history.get_history() == []


def test_changing_capacity_and_turns_matches_fifo_reference():
    randomizer = random.Random(1832)
    history = ChatHistory()
    expected = []
    original_list = history.history
    for number in range(300):
        capacity = randomizer.choice([None, 0, 1, 2, 5, 30, 100])
        history.max_messages = capacity
        if number % 3 == 0:
            history.initialize_turn()
        role = "user" if number % 2 == 0 else "assistant"
        history.add_message(role, Entry(number=number))
        expected.append((role, number, history.current_turn_id))
        if capacity is not None:
            expected = expected[-capacity:] if capacity else []
        assert history.history is original_list
        assert [(message.role, message.content.number, message.turn_id) for message in history.history] == expected
        assert history.get_history() == [
            {"role": role, "content": Entry(number=value).model_dump_json()} for role, value, _ in expected
        ]
