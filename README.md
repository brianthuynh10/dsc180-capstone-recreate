# DSC180 Capstone Project
*Biomarker-Supervised Deep Learning for Pulmonary Edema Assessment from chest X-rays with a focus on interpretability*

## Project Overview

This repository contains a multi-quarter capstone project for DSC180 focused on
biomarker-supervised deep learning for pulmonary edema assessment from chest X-rays.

**Quarter 1** reproduces and evaluates prior work showing that serum biomarkers
(BNP / NT-proBNP) can be used as continuous training labels for convolutional neural
networks, reducing reliance on subjective radiologist annotations. We implement a
reproducible data pipeline, fine-tune ImageNet-pretrained CNNs (VGG variants),
and evaluate model performance using Pearson correlation and MAE.

**Quarter 2** extends this pipeline by focusing on model interpretability. Building on
the biomarker-based regression framework, we aim to combine attribution methods
(e.g., Grad-CAM, saliency maps) with large language models to generate
clinically grounded, human-readable explanations aligned with radiology-report reasoning.

## Repository Structure

- `interpretability/`  
  Folder consisiting of explainability methods implemented that we chose to use for our experiements on CNNs and LLMs
    *`/GradCAM` contains scripts that uses the GradCAM package APIs to help find gradients at selected layers and create visual heatmaps overlaid on a sample image
    *`/imageOcclusion` helps predict log-BNPP values on ablated images and a `AblatedBNPPDataset` to help put into a pytorch data loader

- `notebooks/part1/`  
  Exploratory analysis and modeling notebooks used in Quarter 1,
  including baseline experiments and CNN regression training.

- `notebooks/part2/`  
  Used for analysis an experiementing with LLMs, also used for creating our visualizations incorporated into our presentation poster. 

- `reports/part1/`  
  Quarter 1 submitted report (reproduction and evaluation study).

- `reports/part2/`  
  Quarter 2 proposal and final report (interpretability extension).

-  `models/`
  Models we used for our explainability experiments. Broken into `/MediPhi` and `/CNNs` - where `/MediPhi` contains scripts to help assemble the fine-tuned model and generate labels for an inputted reports. `CNNs` consists of scripts to preprocess image data and train ResNet50 or VGG16 models. 

- `scripts/` 
  Scripts used for running our models and explainability methods. More details of each script will be located in the README.md in this folder. To run a script,
  ```
  python3 -m scripts.<SCRIPT_NAME> # DO NOT INCLUDE .py!
  ```

- `Dockerfile`, `requirements.txt`  
  Environment configuration for reproducible execution.

## Setup & Reproducibility (DSMLP)
The following steps describe how to reproduce training and experiments using the UCSD DSMLP cluster environment.

### Step 0: Prerequisites
Make sure you have Docker Desktop install on your computer. If not, install here for [Mac](https://docs.docker.com/desktop/setup/install/mac-install/) or 
[Windows](https://docs.docker.com/desktop/setup/install/mac-install/](https://docs.docker.com/desktop/setup/install/windows-install/ )

Create a Weights and Biases account ([here](https://wandb.ai/site/)) 
  <ol type="1"> 
    <li> Once you reach the home page, navigate to the top right corner of your profile and click on it</li>
    <li> In the drop down, click on API key where you will be directed to page where you can copy your API key </li>
    <li> Copy the key and place it somewhere you'll remember! You can always navigate back to this page if you lose it</li>
  </ol>
</li>

### Step 1: Cloning the repository
Open and paste the following command
```
git clone https://github.com/brianthuynh10/dsc180-capstone-recreate.git
```

### Step 2: Docker Image Setup: 
Make sure you have Docker open, then navigate to the repository then use the following command in your terminal, 
```
docker buildx build -platform [TARGET_OS]/[TARGET_CPU] -t [DOCKER_USERNAME]/[IMAGE]:[TAG] [CONTEXT_PATH]
```
Example: <br>
<b> NOTE: </b> Since DSMLP does not run using Mac and our data resides on the cluster, we suggest using `linux/amd64` for your ```[TARGET_OS]/[TARGET_CPU]```
```
docker buildx build - platform linux/amd64 -t brianthuynh10/dsc-env:latest .
```
Next, push your image using the following command,
```
docker push [DOCKER_USERNAME]/[IMAGE]:[TAG]
```
Example: 
```
docker push brianthuynh10/dsc-env:latest
```
<b> Make sure your image is public, otherwise DSMLP cannot pull your image! You can change this seting by logging into DockerHub on your browser and changing your image's settings </b> 

### Step 3: DSMLP Cluster: 
First you need to SSH into the DSMLP, then once you're in the jumpbox server run the follwing command to ensure you're using the Docker image you created earlier. Note: You can add the tag `-b` if you want to create a background pod because the model will take a while to train
```
launch.sh \
    -W DSC180A_FA25_A00 -G b1100018875 \
    -i [DOCKER_USERNAME]/[IMAGE]:[TAG] \
    -c [NUMBER_OF_CPUs] -m [SIZE_OF_RAM] -g [NUMBER_OF_GPUs] -v [GPU_VARIANT] \
    -P Always -T -s
```
Example: The configuration below is sufficient to run the model smoothly. Just make sure to have a GPU! You can check whichever one is free [here](https://datahub.ucsd.edu/hub/status)
```
launch.sh \
    -W DSC180A_FA25_A00 -G b1100018875 \
    -i brianthuynh10/dsc-env:latest \
    -c 8 -m 32 -g 1 -v 2080ti \
    -P Always -T -s \
```
### Step 4: Copying the Repo to DSMLP (if you already did that, skip to step 5)
If you cloned the repo to your local machine, you'll have to SCP the repo to DSMLP. First we'll need to zip the entire repository using the command: 
```
zip -r [ZIP_FOLDER_NAME].zip dsc180-capstone-recreate
```
Then use the following command to send it to DSMLP
```
scp <FILE_TO_TRANSFER> [USERNAME]@dsmlp-login.ucsd.edu:<FILE_NAME_FOR_DSMLP>
```
You should see the file appear in your private folder in DSMLP where you can go into the DSMLP terminal to unzip using
```
unzip [ZIP_FOLDER_NAME].ZIP
```
### Step 5: Running the Script
Inside DSMLP, navigate to path of the repo (if you didn't change any names the directory should be `private/dsc180-capstone-recreate`). From there you can run the command,
```
python3 main.py all > log.txt &
```
At some point early in the run, you'll be prompted to enter your W&B API key which will allow you to see training status and the results on the test set. There will also be a `log.txt` file training progress and an output folder the saves the best model incase the process crashes during training.  Once the process is over, you'll be able to see graphs of model's predicitons on the test set and at each step on the validation set. 

## Notes on Data and Usage

This repository is intended for academic coursework only.
The underlying dataset is subject to institutional restrictions and is not publicly redistributable.
