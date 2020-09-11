#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 13:02:36 2020

@author: safir
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="classic_ID3_DecisionTree", 
    version="2.0.2",
    author="Safir Motiwala",
    author_email="safirmotiwala@gmail.com",
    description="ID3 is a Machine Learning Decision Tree Classification Algorithm that uses two methods to build the model. The two methods are Information Gain and Gini Index.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/safir72347/ML-ID3-Decision-Tree-Classification-Library-PyPi",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)