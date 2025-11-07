import argparse
from src.clean_data import clean_main
from src.train import Trainer

def main(): 
     # -- Pull Data -- 
    train, val, test = clean_main() # XRayDataset objects

    # -- Train Model (using papers setup) -- 
    trainer = Trainer(epochs=50, lr=1e-5, batch_size=16)
    trainer.create_dataloaders(train, val, test)
    trainer.train()

    # -- Evaluate Model --
    trainer.evaluate()

if __name__ == "__main__":
    main()




