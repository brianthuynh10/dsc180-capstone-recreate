# src/__init__.py
from .clean_data import main as clean 
from .train import Trainer

__all__ = ["clean", "Trainer"]
