import subprocess

subprocess.run(["isort"])
subprocess.run(["black", "ganji", "tests", "scripts"])
