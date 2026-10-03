from dataclasses import dataclass


@dataclass
class Event:
    t_start: float
    t_end: float
    kind: str = "wait"


def _advance(cursor, index, local_time, events, dt):
    event = events[index]
    target = min(cursor + dt, event.t_end)
    local_target = target - event.t_start
    step = local_target - local_time
    cursor = target
    local_time = local_target
    if target >= event.t_end - 1e-9:
        index += 1
        local_time = 0.0
    return cursor, index, local_time, step


def test_replay_transitions_to_every_later_event():
    events = [
        Event(0.0, 2.0),
        Event(2.0, 4.0, "animation"),
        Event(4.0, 6.0),
    ]
    cursor = 1.0
    index = 0
    local_time = 1.0
    visited = []
    while index < len(events):
        visited.append(index)
        cursor, index, local_time, _ = _advance(
            cursor, index, local_time, events, 0.75
        )
    assert visited == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_replay_event_transition_reinitializes_next_animation():
    initialized = {0}
    index = 0
    events = [Event(0.0, 1.0), Event(1.0, 2.0, "animation")]
    cursor = 0.5
    local_time = 0.5
    while index < len(events):
        cursor, index, local_time, _ = _advance(
            cursor, index, local_time, events, 0.5
        )
        if index < len(events) and index not in initialized:
            initialized.add(index)
    assert initialized == {0, 1}
