"""Manipulate files of GANJI project."""

import os
import sys
from typing import Optional, Tuple

from serde.json import from_json, to_json
from serde.toml import from_toml, to_toml

from ganji.project.model import Config, State, _initial_state


def _config_path(dir: str):
    return os.path.join(dir, "ganji.toml")


def _dump_config(dir: str, config: Config):
    s = to_toml(config)
    with open(_config_path(dir), "w") as config_file:
        config_file.write(s)


def _load_config(dir: str) -> Config:
    with open(_config_path(dir)) as config_file:
        return from_toml(Config, config_file.read())


def _state_path(dir: str):
    return os.path.join(dir, "state.json")


def dump_state(dir: str, state: State):
    s = to_json(state)
    with open(_state_path(dir), "w") as state_file:
        state_file.write(s)


def _load_state(dir: str) -> Optional[State]:
    state_path = _state_path(dir)
    if os.path.exists(state_path):
        with open(state_path) as state_file:
            return from_json(State, state_file.read())
    else:
        return None


def load_metadata(dir: str) -> Tuple[Config, State]:
    config = _load_config(dir)
    state = _load_state(dir)
    if state is None:
        state = _initial_state(config)
    else:
        state.config = config
    return (config, state)


def new(dir: str, config: Config):
    if os.path.exists(dir):
        print(f"already exists: {dir}", file=sys.stderr)
        exit(1)
    os.mkdir(dir)
    init(dir, config)


def init(dir: str, config: Config):
    config_path = _config_path(dir)
    if os.path.exists(config_path):
        print(f"already exists: {config_path}", file=sys.stderr)
        exit(1)
    _dump_config(dir, config)
