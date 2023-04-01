#!/usr/bin/env python3
import subprocess

subprocess.run(["pflake8", "ganji", "tests", "scripts"])
