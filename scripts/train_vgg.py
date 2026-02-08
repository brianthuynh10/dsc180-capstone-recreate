from models.vgg.train import Trainer
from models.vgg.data import main as clean


def main(): 
     # -- Pull Data -- 
    print('Beginning Data Cleaning')
    train, val, test, y_mean, y_std = clean() # XRayDataset objects

    # -- Train Model (using papers setup) -- 
    print('Model created & training will start now')
    trainer = Trainer(epochs=50, lr=1e-5, batch_size=16, train_dataset = train, val_dataset = val, test_dataset = test, train_mean=y_mean, train_std=y_std)
    # -- This will handle the training loop, validation, and testing --
    trainer.train()
    

if __name__ == "__main__":
    main()

