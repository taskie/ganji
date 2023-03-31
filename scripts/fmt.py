#!/usr/bin/env python3
import subprocess

subprocess.run(["isort", "ganji", "tests", "scripts"])
subprocess.run(["black", "ganji", "tests", "scripts"])
