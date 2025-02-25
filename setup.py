# This project is licensed under the Creative Commons Attribution 4.0 International License.
# See https://creativecommons.org/licenses/by/4.0/ for details.

from setuptools import setup, find_namespace_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name = "omniparser",
    version = "0.1.0",
    author = "Microsoft Research AIF Frontiers",
    description = 'OmniParser, comprehensive method for parsing user interface screenshots into structured and easy-to-understand elements',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    include_package_data = True,
    packages = find_namespace_packages(where='omniparser'),
    package_dir = {"omniparser": "omniparser"},
    install_requires = [
        "opencv-python-headless",
        "gradio",
        "dill",
        "accelerate",
        "timm",
        "einops==0.8.0",
        "paddlepaddle",
        "paddleocr",
    ],
    classifiers=[
        "License :: CC-BY-4.0"
    ],
    license_files = ['LICENSE']
)