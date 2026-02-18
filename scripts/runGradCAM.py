from interpretability.GradCAM.data import create_groups
from interpretability.GradCAM.GradCAM import GradCAMWrapper
from models.CNNs.assembleCNN import build_grayscale_cnn
from interpretability.GradCAM.BNPPDataset import BNPPDataset
import os 
from torch.utils.data import DataLoader
import pandas as pd
import torch
import matplotlib.pyplot as plt
import ast

def batch_predict(model, df, batch_size=16, device="cuda"):
    df = df.reset_index(drop=True)
    train_df = pd.read_csv('/home/bth001/teams/b1/BNPP_DT_val_with_ages.csv')
    avg = train_df['bnpp_value_log'].mean()
    std = train_df['bnpp_value_log'].std()

    base = os.path.expanduser('~/teams/b1/')
    model.eval().to(device)

    dataset = BNPPDataset(df, base)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,      
        pin_memory=True     
    )

    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(device)        # (B, 1, H, W)
            out = model(imgs).squeeze(0)   # (B,)
            out = out * std + avg
            for i, pred in zip(idxs, out.cpu().numpy()):
                df.at[i.item(), "bnpp_prediction"] = pred.item()
    return df

def main(): 
    print('Starting Phase 1: Loading in data and model...')

    full_reports = pd.read_csv(
        "~/teams/b1/processed_data/connected_bnpp_reports.csv"
    )
    full_reports.dropna(subset=['hdf5_file_name'], inplace=True)
    full_reports['hdf5_file_name'] = full_reports['hdf5_file_name'].apply(ast.literal_eval)

    model = build_grayscale_cnn(model_name="resnet50")
    grouped_dict = create_groups(full_reports)

    print('Phase 1 Done')
    print('Starting Phase 2: Generating predictions per group')
    print(50 * '=')

    target_layer = model.layer4[-1]

    for group_name, group_df in grouped_dict.items():

        print(f'Generating Predictions for Group: {group_name}')

        if group_name == 'absent':
            group_df = group_df.sample(2000)

        group_df = batch_predict(model, group_df, batch_size=32)
        grouped_dict[group_name] = group_df

        print(f'Finished predictions for {group_name}')
        print(f'Building GradCAM map for {group_name} at layer4[-1]')

        cam = GradCAMWrapper(
            model=model,
            target_layer=target_layer
        )

        average_cam_map = cam.generate_average_cam(group_df)
        heatmap = cam.build_averaged_cam_map(average_cam_map, group_df)

        # Output folder
        out_dir = "outputs/resnet_last_layer"
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{group_name}_cam.png")
        plt.imsave(out_path, heatmap)

        print(f"GradCAM image saved at {out_path}")
        print(50 * '=')

if __name__ == "__main__":
    main()