from setuptools import setup
import os

os.makedirs("shared")
with open("shared/__init__.py", "w") as out:
    with open("shared.py") as src:
        out.write(src.read())

setup(
    name="CS451Practicals",
    py_modules=["shared"],
    long_description=open("README.md").read(),
    dependencies=[l.strip() for l in open("requirements.txt")],
)