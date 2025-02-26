Drift Detection Module

This repository contains a drift detection module that helps monitor data distribution changes over time using Evidently and other statistical tools.

Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

To verify your Python installation, run:

**python --version**

If Python is not installed, download and install it from the official Python website.

Setup Instructions

1. Create a Virtual Environment

To keep dependencies isolated, create a virtual environment:

python -m venv drift_detection_env

Activate the environment:

Mac/Linux:

**source drift_detection_env/bin/activate**

2. Install Required Libraries

Upgrade pip and setuptools, then install the required dependencies:

**pip install -U pip setuptools
pip install evidently numpy pandas scipy matplotlib seaborn**

3. Install Additional Libraries

For extended functionalities like visualization and model evaluation, install:

**pip install jupyterlab scikit-learn**

4. Verify Installation

Ensure that all libraries are installed correctly by running:

**python -c "import evidently, numpy, pandas, scipy; print('All libraries imported successfully!')"**

5. Run the Initial Script

To test the setup, execute the main script: **python3 main.py**

To change dataset do modify
**Reference Data**: A dataset representing the baseline distribution.
**Current Data**: A dataset to compare against the reference.
