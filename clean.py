from src import clean

train, val, test, y_mean, y_std = clean()

import torch

os.makedirs("data", exist_ok=True)

torch.save({
    "train": train,
    "val": val,
    "test": test,
    "y_mean": y_mean,
    "y_std": y_std,
}, "data/cleaned_data.pt")

print("Saved cleaned dataset + stats!")
