# VIME-Pytorch
Pytorch implementation of ["VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain"](https://proceedings.neurips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf)

Official implementation in tensorflow: https://github.com/jsyoon0823/VIME

## Code explanation
1. data_loader.py
   - Load and preprocess MNIST and other tabular data
2. model.py
   - Models required for VIME training
3. utils.py
   - Some utility functions for metrics and VIME frameworks.
4. train.py
   - Modules that include training, testing, etc.
5. main.py
   - Adjusting hyperparameters using hydra
6. conf/config.yaml
   - File to adjust hyperparameters


## Requirement
- Python >= 3.6

```bash
python -m venv venv
. venv/bin/activate
```

## Installation
```bash
pip install -r requirements.txt
```

## Command
```bash
python main.py
```