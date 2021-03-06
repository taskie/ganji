"""Metadata schema."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from serde import deserialize, serialize


@deserialize
@serialize
@dataclass
class Config:
    mode: Optional[str]
    batch_size: int
    codepoint_set: str
    epoch_end: int
    font: str
    font_index: Optional[int]
    unit: int
    density_quantile_min: Optional[float]
    density_quantile_max: Optional[float]
    dataset_random_seed: Optional[int]


@deserialize
@serialize
@dataclass
class State:
    epoch: int
    update_time: Optional[float]
    config: Optional[Config]


def _initial_state(config: Config) -> State:
    return State(epoch=0, update_time=datetime.now().timestamp(), config=config)
