import torch
import pandas as pd
from models.CNNs.assembleCNN import build_grayscale_cnn
from interpretability.imageOcclusion.ablationPredict import run_ablation_patch_across_image
    
def main():
    MODEL_NAME = "vgg16"
    print('Running runAblation.py!')
    predictions_df = pd.read_csv(
        "~/private/dsc180-capstone_copy_1/notebooks/vgg16_outputs_no_ablation.csv"
    )
    predictions_df = predictions_df.set_index("unique_key")

    resnet_model = build_grayscale_cnn(model_name=MODEL_NAME)
    
    run_ablation_patch_across_image(resnet_model, predictions_df, MODEL_NAME)
    print('Script is done Running!')


if __name__ == "__main__":
    main()