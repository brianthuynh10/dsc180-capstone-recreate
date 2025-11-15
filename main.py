from src import Trainer, clean
import wandb

def main(): 
    # Initialize W&B (sweep agent injects config here)
    wandb.init(project="dsc180-capstone-recreate")

    # ---- Data Loading ----
    train, val, test, y_mean, y_std = clean()
    print("Data loaded successfully.")

    # Read hyperparameters from sweep config
    batch_size = [8, 16, 32, 64, 128]
    seeds = [0, 1, 2, 3, 4]
    lr    =  1e-5
    epochs = 10
    # ---- Hyperparameter Sweep Loop ----
    for bs in batch_size:
        for seed in seeds:
            if (bs == 8 and seed !=4) or (bs == 16 and seed == 0): continue
            else: 
                # ---- Trainer ----
                print(f"Starting training with batch size {bs} and seed {seed}")
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
