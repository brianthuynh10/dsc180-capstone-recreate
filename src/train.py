# src/train.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.clean_data import main as clean_main
from src.model import create_model
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import wandb
import os


class Trainer:
    def __init__(
        self,
        epochs=50,
        lr=1e-5,
        batch_size=16,
        project="vgg16-xray-regression",
        run_name=None,
        device=None,
    ):
        """Initialize trainer, model, optimizer, and wandb run."""
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🖥️ Using device: {self.device}")
        print(f"📦 Hyperparameters | Epochs: {self.epochs}, LR: {self.lr}, Batch Size: {self.batch_size}")

        # --- Initialize model, loss, and optimizer ---
        self.model = create_model().to(self.device)
        self.criterion = nn.L1Loss()  # MAE loss
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.lr)
        self.best_val_loss = float("inf")

        # --- Create output directory ---
        os.makedirs("outputs", exist_ok=True)

        # --- Initialize W&B ---
        wandb.init(project=project, name=run_name, config={
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "loss": "MAE"
        })
        wandb.watch(self.model, log="all", log_freq=100)

    def load_data(self):
        """Load cleaned data and create dataloaders."""
        print("📦 Loading datasets...")
        train_dataset, val_dataset, test_dataset = clean_main()
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def train(self):
        """Main training loop."""
        self.load_data()
        print("Beginning training...")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss_total = 0.0
            all_train_preds, all_train_labels = [], []

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels.view(-1, 1))
                loss.backward()
                self.optimizer.step()

                train_loss_total += loss.item()
                all_train_preds.append(outputs.detach().cpu())
                all_train_labels.append(labels.detach().cpu())

            avg_train_loss = train_loss_total / len(self.train_loader)
            train_r = self._pearson_corr(all_train_preds, all_train_labels)

            avg_val_loss, val_r, val_fig = self.validate(epoch)
            self._save_best_model(avg_val_loss)

            # --- Log to W&B ---
            wandb.log({
                "epoch": epoch + 1,
                "train_mae": avg_train_loss,
                "train_r": train_r,
                "val_mae": avg_val_loss,
                "val_r": val_r,
                "val_scatter": wandb.Image(val_fig)
            })
            plt.close(val_fig)

            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train MAE: {avg_train_loss:.4f} | Train r: {train_r:.4f} | "
                  f"Val MAE: {avg_val_loss:.4f} | Val r: {val_r:.4f}")

        wandb.finish()
        print("✅ Training complete.")

    def validate(self, epoch):
        """Run validation and save scatter plot."""
        self.model.eval()
        val_loss_total = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                val_loss_total += self.criterion(outputs, labels.view(-1, 1)).item()
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = val_loss_total / len(self.val_loader)
        val_r = self._pearson_corr(all_preds, all_labels)

        # --- Create scatter plot ---
        fig, ax = plt.subplots()
        ax.scatter(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(), alpha=0.5)
        ax.set_xlabel("True BNP (log)")
        ax.set_ylabel("Predicted BNP (log)")
        ax.set_title(f"Epoch {epoch+1}: MAE={avg_val_loss:.4f}, r={val_r:.4f}")

        plt.savefig(f"outputs/scatter_epoch_{epoch+1}.png")

        return avg_val_loss, val_r, fig

    def evaluate(self):
        """Evaluate best model on test set."""
        print("🔍 Evaluating best model on test set...")
        self.model.load_state_dict(torch.load("outputs/best_model.pt", map_location=self.device))
        self.model.eval()

        test_loss_total = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                test_loss_total += self.criterion(outputs, labels.view(-1, 1)).item()
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())

        avg_test_loss = test_loss_total / len(self.test_loader)
        test_r = self._pearson_corr(all_preds, all_labels)
        wandb.log({"test_mae": avg_test_loss, "test_r": test_r})
        print(f"Test MAE: {avg_test_loss:.4f} | 🔗 Test r: {test_r:.4f}")
        return avg_test_loss, test_r

    def _pearson_corr(self, preds, labels):
        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        return pearsonr(labels.flatten(), preds.flatten())[0]

    def _save_best_model(self, avg_val_loss):
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            torch.save(self.model.state_dict(), "outputs/best_model.pt")
            print("Saved new best model!")
