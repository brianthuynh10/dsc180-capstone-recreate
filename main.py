from src import Trainer, clean
import wandb
import torch

def main(): 
    # Initialize W&B (sweep agent injects config here)
    wandb.init(project="vgg16-xray-regression")

    # ---- Data Loading ----
    train, val, test, y_mean, y_std = clean()
    print("Data loaded successfully.")

    # Read hyperparameters from sweep config
    bs    = wandb.config.batch_size
    lr    = wandb.config.lr
    epochs = wandb.config.epochs
    seed   = wandb.config.seed

    # ---- Trainer ----
    trainer = Trainer(
        epochs=epochs,
        lr=lr,
        batch_size=bs,
        seed=seed,
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        train_mean=y_mean,
        train_std=y_std,
        run_name=f"bs{bs}-seed{seed}"
    )

    wandb.watch(trainer.model, log="all", log_freq=100)

    trainer.train()
    trainer.evaluate()
    wandb.finish()

if __name__ == "__main__":
    main()
