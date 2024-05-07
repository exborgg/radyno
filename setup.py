# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 22:33:21 2024

@author: Andrea
"""

from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.4'
DESCRIPTION = 'Particles tracking and radiation spectrum evaluation'

# Setting up
setup(
    name="radyno",
    version=VERSION,
    author="exborgg (Andrea Frazzitta)",
    author_email="<andrea.frazzitta@uniroma1.it>",
    description=DESCRIPTION,
    packages=find_packages(),
    #install_requires=['opencv-python', 'pyautogui', 'pyaudio'],
    keywords=['python', 'beam', 'radiation', 'spectrum', 'mpi', 'parallel'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)