import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.clean_data import main as clean_main
from src.model import make_vgg16_model as create_model
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import wandb
import os
import numpy as np
from sklearn.linear_model import LinearRegression

class Trainer:
    def __init__(
        self,
        epochs=50,
        lr=1e-5,
        batch_size=16,
        train_mean: float=None,
        train_std: float=None,
        project="vgg16-xray-regression",
        run_name=None,
        device=None,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        seed=42
        ):

        """Set random seeds for reproducibility."""
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        """Initialize trainer, model, optimizer, and wandb run."""
        self.train_mean = train_mean
        self.train_std = train_std
        self.model = create_model()
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.L1Loss()
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.best_val_loss = float("inf")
        self.project = project
        self.run_name = run_name
        self.seed = seed
    
        print(f"Using device: {self.device}")
        print(f"Hyperparameters | Epochs: {self.epochs}, LR: {self.lr}, Batch Size: {self.batch_size}")

        # Move model to gpu:
        self.model.to(self.device)

        # --- Create output directory ---
        os.makedirs("outputs", exist_ok=True)


    def create_dataloaders(self):
        """Load cleaned data and create dataloaders."""
        # Every DataLoader uses its own generator
        g = torch.Generator()
        g.manual_seed(self.seed)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
            generator=g
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
            generator=g
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
            generator=g
        )

    def train(self):
        """Main training loop."""
        self.create_dataloaders()
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

        print("Training complete.")

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

        # Convert back to BNPP value, not log
        avg_val_loss = val_loss_total / len(self.val_loader)
        val_r = self._pearson_corr(all_preds, all_labels)

        # standardized values
        pred_standard = torch.cat(all_preds).numpy().flatten()
        labels_standard = torch.cat(all_labels).numpy().flatten()

        # undo standardization
        preds_log = (pred_standard * self.train_std) + self.train_mean
        labels_log = (labels_standard * self.train_std) + self.train_mean
        
        # true scale 
        preds_real = 10 ** (preds_log-1)
        labels_real = 10 ** (labels_log-1)

        # --- Create scatter plot in original BNP scale ---
        fig, ax = plt.subplots()
        ax.scatter(labels_real, preds_real, alpha=0.5, label="Predictions")
        
        # --- Fit line (y = m*x + b) in log10 space for stability ---
        log_true = np.log10(labels_real + 1e-8).reshape(-1, 1)
        log_pred = np.log10(preds_real + 1e-8)
        reg = LinearRegression().fit(log_true, log_pred)
        reg_line = 10 ** reg.predict(log_true)
        ax.plot(labels_real, reg_line, 'r-', alpha=0.75, label='Regression line')

        
        # Reference y = x (perfect prediction)
        lims = [min(labels_real.min(), preds_real.min()), max(labels_real.max(), preds_real.max())]
        ax.plot(lims, lims, 'k--', alpha=0.75, label='y=x')
        
        # --- Labels, scales, legend ---
        ax.set_xlabel("Actual NT-proBNP")
        ax.set_ylabel("Predicted NT-proBNP")
        ax.set_title("Predicted vs Actual NT-proBNP")
        ax.legend()

               # --- Log scale ---
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        # --- Major ticks at powers of 10 ---
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        
        # --- Label only the powers of 10 (10, 100, 1000, etc.) ---
        ax.xaxis.set_major_formatter(ticker.LogFormatter(base=10.0, labelOnlyBase=True))
        ax.yaxis.set_major_formatter(ticker.LogFormatter(base=10.0, labelOnlyBase=True))

        return avg_val_loss, val_r, fig

    def evaluate(self):
        """Evaluate best model on test set and log results + scatter plot."""
        print("🔍 Evaluating best model on test set...")
    
        # --- Load best model ---
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
    
        preds_standard = torch.cat(all_preds).numpy().flatten()
        labels_standard = torch.cat(all_labels).numpy().flatten()
        # undo standardization

        preds_log = (preds_standard * self.train_std) + self.train_mean
        labels_log = (labels_standard * self.train_std) + self.train_mean

        preds_real = 10 ** (preds_log-1)
        labels_real = 10 ** (labels_log-1)  
        # --- Scatter plot ---
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(labels_real, preds_real, alpha=0.4, color="steelblue", s=20, label="Predictions")
    
        # Regression line (log10 space)
        log_true = np.log10(labels_real + 1e-8).reshape(-1, 1)
        log_pred = np.log10(preds_real + 1e-8)
        reg = LinearRegression().fit(log_true, log_pred)
        x_range = np.linspace(log_true.min(), log_true.max(), 100).reshape(-1, 1)
        y_range = reg.predict(x_range)
        ax.plot(10 ** x_range, 10 ** y_range, color='red', linewidth=2, label='Regression line')
    
        # Reference y=x
        lims = [min(labels_real.min(), preds_real.min()), max(labels_real.max(), preds_real.max())]
        ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction')
    
        # Axis setup
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(ticker.LogFormatter(base=10.0, labelOnlyBase=True))
        ax.yaxis.set_major_formatter(ticker.LogFormatter(base=10.0, labelOnlyBase=True))
        ax.set_xlabel("Actual NT-proBNP")
        ax.set_ylabel("Predicted NT-proBNP")
        ax.set_title(f"Predicted vs Actual NT-proBNP (r = {test_r:.3f})")
        ax.legend(frameon=False)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        plt.tight_layout()
    
        # --- Log results ---
        wandb.log({
            "test_mae": avg_test_loss,
            "test_r": test_r,
            "test_scatter": wandb.Image(fig)
        })
        plt.close(fig)
    
        print(f"Test MAE: {avg_test_loss:.4f} | Test r: {test_r:.4f}")
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
