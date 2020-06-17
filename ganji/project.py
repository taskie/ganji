"""Manipulate GANJI project."""

import json
import os
import sys

import toml


def _props_path(dir):
    return os.path.join(dir, "ganji.toml")


def _state_path(dir):
    return os.path.join(dir, "state.json")


def _dump_metadata(dir, obj):
    with open(_props_path(dir), "w") as props_file:
        toml.dump(obj["props"], props_file)
    state = obj.get("state")
    if state is not None:
        with open(_state_path(dir), "w") as state_file:
            json.dump(state, state_file, ensure_ascii=False)


def _load_metadata(dir):
    obj = {}
    with open(_props_path(dir)) as props_file:
        obj["props"] = toml.load(props_file)
    with open(_state_path(dir)) as state_file:
        obj["state"] = json.load(state_file)
    return obj


def new(dir, obj):
    if os.path.exists(dir):
        print(f"already exists: {dir}", file=sys.stderr)
        exit(1)
    os.mkdir(dir)
    obj["props"]["font"] = os.path.abspath(obj["props"]["font"])
    init(dir, obj)


def init(dir, obj):
    json_path = os.path.join(dir, "ganji.toml")
    if os.path.exists(json_path):
        print(f"already exists: {json_path}", file=sys.stderr)
        exit(1)
    _dump_metadata(dir, obj)
