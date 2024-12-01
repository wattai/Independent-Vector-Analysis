# WIP: Independent Vector Analysis; IVA

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This repository aims to provide tools and resources for implementing IVA algorithms and applications.

![IVA Example Image](https://wallpapers.miyanova.com/data/367_600.jpg)

## Setup

1. Clone the repository

    ```shell
    git clone git@github.com:wattai/Independent-Vector-Analysis.git
    cd Independent-Vector-Analysis
    ```

1. Install dependencies

    ```shell
    pip install -e .
    ```

## Run the IVA algorithm

1. Run a demo script

    ```shell
    python demo_script.py
    ```

## What is Independent Vector Analysis?

Independent Vector Analysis (IVA) is a computational technique used in signal processing to separate mixed signals into their original, independent components. It is an extension of Independent Component Analysis (ICA) that is particularly useful when dealing with multiple datasets or multidimensional data. IVA assumes that the source signals are statistically independent and aims to maximize this independence to achieve separation. This method is widely applied in fields such as biomedical signal processing, audio source separation, and telecommunications. IVA is advantageous over ICA when the datasets have dependencies across different dimensions, as it can exploit these dependencies to improve the separation performance. The technique often involves optimization algorithms and requires careful consideration of the model parameters to ensure accurate results. IVA's effectiveness can be influenced by the choice of cost functions and constraints, which are crucial for capturing the statistical properties of the source signals. Additionally, the performance of IVA can be enhanced by incorporating prior knowledge about the signal structure or by using advanced algorithms that adaptively adjust to the data characteristics.
