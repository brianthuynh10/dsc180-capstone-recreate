from src import clean, Trainer
import wandb

def main(): 
    # Initialize W&B (sweep agent injects config here)
    wandb.init(project="vgg16-xray-regression")

    # -- Pull Data -- 
    print('Beginning Data Cleaning')
    train, val, test = clean()  # XRayDataset objects

    # Read hyperparameters from sweep config
    bs = wandb.config.batch_size
    lr = wandb.config.lr
    epochs = wandb.config.epochs
    seed = wandb.config.seed

    # -- Create Trainer --
    print('Model created & training will start now')

    trainer = Trainer(
        epochs=epochs,
        lr=lr,
        batch_size=bs,
        seed=seed,
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        run_name=f"batch-size-{bs}-seed{seed}"
    )

    # Track gradients & parameters
    wandb.watch(trainer.model, log="all", log_freq=100)

    # Train + evaluate (same run)
    trainer.train()
    trainer.evaluate()

    # End run cleanly
    wandb.finish()

if __name__ == "__main__":
    main()
