#!/usr/bin/env python3
import subprocess

subprocess.run(["coverage", "run", "-p", "-m", "pytest", "tests"])
subprocess.run(["coverage", "combine"])
subprocess.run(["coverage", "report"])
subprocess.run(["coverage", "html"])
