from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from serde import deserialize, serialize


@deserialize
@serialize
@dataclass
class Config:
    batch_size: int
    codepoint_set: str
    epoch_end: int
    font: str
    unit: int


@deserialize
@serialize
@dataclass
class State:
    epoch: int
    update_time: Optional[float]
    config: Optional[Config]


def _initial_state(config: Config) -> State:
    return State(epoch=0, update_time=datetime.now().timestamp(), config=config)
