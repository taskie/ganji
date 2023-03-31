#!/usr/bin/env python3
import subprocess

subprocess.run(["flake8", "ganji", "tests", "scripts"])
