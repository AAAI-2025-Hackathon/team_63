import os
import torch
import random
import numpy as np

class Logger:
    """
    Simple logger to track training/evaluation metrics.
    In practice, you might use TensorBoard or a more sophisticated approach.
    """

    def __init__(self, log_path=None):
        self.log_path = log_path
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.logs = []

    def log(self, message):
        print(message)
        self.logs.append(message)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(message + "\n")

def save_model(model, path):
    """
    Save PyTorch or stable-baselines3 model to path.
    For stable-baselines3, use model.save(path).
    For PyTorch modules, use torch.save().
    """
    # If it's a stable-baselines3 model, it has a 'save' attribute
    if hasattr(model, "save"):
        model.save(path)
    else:
        torch.save(model.state_dict(), path)

def load_model(model_class_or_sb, path, **kwargs):
    """
    Load a model from the specified path. If model_class_or_sb
    is a stable-baselines3 class, call .load(). Otherwise,
    treat it as a PyTorch module.
    """
    # If it's a stable-baselines3 class
    if hasattr(model_class_or_sb, "load"):
        return model_class_or_sb.load(path, **kwargs)
    else:
        # Assume it's a PyTorch nn.Module
        model = model_class_or_sb(**kwargs)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
