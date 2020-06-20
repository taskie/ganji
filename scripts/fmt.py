import subprocess

subprocess.run(["isort", "-y"])
subprocess.run(["black", "ganji", "tests", "scripts"])
