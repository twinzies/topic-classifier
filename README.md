# Neural Network Topic Classifier Project

## Overview

This project showcases two implementations of a neural network classifier:

- **From scratch (NumPy-only):** Located in `notebooks/numpy_topic_classifier.ipynb`, illustrating the core mechanics without deep learning libraries.
- **Modular PyTorch version:** Structured for scalability and to demonstrate standard practice, with clear architecture, training, and visualization modules.

## Features

- Clean modular design (`model/`, `visualization/`)
- Tests included (using `logging`)
- Example shell script for easy execution
- Author information embedded in each module

## Setup

```bash
pip install -r requirements.txt

your-project/
├── data/
│   └── README.md
├── model/
│   ├── architecture.py    PyTorch architecture
│   └── train.py           Training and evaluation logic
├── visualization/
│   └── plots.py           Plotting utilities
├── notebooks/
│   └── numpy_topic_classifier.ipynb       A NumPy notebook implementation.
├── tests.py
├── main.py
├── run.sh
├── requirements.txt
├── README.md
├── .gitignore
