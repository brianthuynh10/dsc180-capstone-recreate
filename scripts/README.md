# Scripts

This folder contains all runnable pipelines for our DSC180 Capstone experiments, including:

- CNN training (VGG16 and ResNet50)
- Image ablation experiments
- Grad-CAM interpretability analysis
- Inference using our fine-tuned MediPhi model

Each script is designed to be executed as a Python module from the root of the repository.

---

## Getting Started

Before running any scripts:

1. Navigate to the root of the project directory.  
   Your terminal should look something like:

   ```
   private/dsc180-capstone/
   ```

2. Run scripts as modules using the `-m` flag.  
   Do **not** include the `.py` extension.

### General Command Format

```bash
python3 -m scripts.<script_name>
```

### Example

```bash
python3 -m scripts.runAblation
```

---

## 📂 Script Descriptions

---

### `mediphi.py`

Assembles and runs inference using our fine-tuned Microsoft MediPhi model trained on radiology reports.

**What it does:**
- Loads the fine-tuned MediPhi checkpoint
- Runs predictions on a small batch of example reports included in the repository
- Outputs model predictions for inspection

**Use case:**  
Quick validation of the fine-tuned language model on report-level data.

---

### `runAblation.py`

Runs our image ablation experiment on the test set.

**What it does:**
- Applies localized patch ablations to X-ray images
- Runs ablated images through either VGG16 or ResNet50
- Records predictions for each ablated image
- Saves outputs as a CSV file (including image ID and predicted value)

**Use case:**  
Quantifying model sensitivity to specific spatial regions of the image.

---

### `runGradCAM.py`

Runs the Grad-CAM interpretability pipeline across grouped patient categories.

**What it does:**
- Groups patients by edema severity
- Runs Grad-CAM on each group
- Averages gradient maps across the group
- Converts averaged gradients into heatmaps
- Overlays heatmaps onto representative X-ray images
- Saves visualizations to the `outputs/` directory

**Use case:**  
Understanding where CNN models focus when making regression predictions.

---

### `train_resnet.py`

Trains a ResNet50 model for X-ray image regression (predicting log-BNPP).

**What it does:**
- Loads training and validation data
- Trains ResNet50 for regression
- Logs training metrics to Weights & Biases
- Saves:
  - Latest checkpoint (after each epoch)
  - Best-performing model (based on validation metric)
- Outputs saved under the designated `outputs/` folder

**Important:**  
Training may take 1–2 hours depending on GPU availability.  
We recommend running this in a background pod (e.g., DSMLP GPU instance).

---

### `train_vgg.py`

Trains a VGG16 model for X-ray image regression (predicting log-BNPP).

**What it does:**
- Same pipeline structure as `train_resnet.py`
- Uses VGG16 backbone instead of ResNet50
- Logs metrics to Weights & Biases
- Saves latest and best-performing model checkpoints

**Important:**  
Also requires GPU resources and may take 1–2 hours.

