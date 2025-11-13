import pandas as pd 
import numpy as np
import h5py
from torchvision import transforms
from src.XRayDataset import XRayDataset
import os 

def grab_keys(path, lst): 
    with h5py.File(path, 'r') as f:
        lst.extend(list(f.keys()))

def find_data_from_dataframe(img_keys, train_df, val_df, test_df):
    """
    Keep the keys that are in the train_df, val_df, or test_df
    """
    train_df = train_df.loc[train_df.index.intersection(img_keys)]
    val_df = val_df.loc[val_df.index.intersection(img_keys)]
    test_df = test_df.loc[test_df.index.intersection(img_keys)]

    return train_df, val_df, test_df

def shrink_image(arr):
    """
    Shrinks a 1024x1024 image to 256x256 by averaging 4x4 blocks
    """
    return arr.reshape(256, 4, 256, 4).mean(axis=(1, 3))

def add_image_data_df(df, file_paths):
    """
    Adds image data from HDF5 files to the DataFrame based on the 'id' index.
    """
    id_set = set(df.index)
    all_images = {}

    for path in file_paths:
        try:
            with h5py.File(path, "r") as file:
                # Iterate only over keys that are in df['id']
                for key in set(file.keys()) & id_set:
                    arr = file[key][()]
                    if isinstance(arr, np.ndarray) and arr.size > 0:
                        if shrink_image is not None:
                            arr = shrink_image(arr)
                        all_images[key] = arr
        except OSError as e:
            print(f"Failed to read {path}: {e}")

    # Map images back to the DataFrame
    df["img_arr"] = df.index.map(all_images).fillna(np.nan)

    return df

def main():
    """ Cleans the data and returns train, val, test datasets """
    # HDF5 Paths:
    home_dir = os.path.expanduser("~")
    base_path = os.path.join(home_dir, "teams", "b1")
    hdf5_paths = [os.path.join(base_path, "bnpp_frontalonly_1024_0_1.hdf5")] + [
         os.path.join(base_path, f"bnpp_frontalonly_1024_{i}.hdf5") for i in range(1, 11)
    ]
    # CSV Files:
    train_data = os.path.join(base_path, 'BNPP_DT_train_with_ages.csv')
    val_data = os.path.join(base_path, 'BNPP_DT_val_with_ages.csv')
    test_data = os.path.join(base_path, 'BNPP_DT_test_with_ages.csv')

    # 1: img_keys list
    print('Clean Data Step 1')
    img_keys = []
    for path in hdf5_paths: 
        grab_keys(path, img_keys)

    # 2: read csv files
    print('Clean Data Step 2')
    train_df, val_df, test_df = pd.read_csv(train_data), pd.read_csv(val_data), pd.read_csv(test_data)
    desired_cols = train_df.columns

    # consistency in columns
    
    train_df = train_df[desired_cols].set_index('unique_key')
    val_df = val_df[desired_cols].set_index('unique_key')
    test_df = test_df[desired_cols].set_index('unique_key')

    # 3: find the data we need to keep (originally used when we had little data)
    print('Clean Data Step 3')
    train_df, val_df, test_df = find_data_from_dataframe(img_keys, train_df, val_df, test_df)

    # 4: add image data to dataframe
    print('Clean Data Step 4')
    train_df = add_image_data_df(train_df, hdf5_paths)
    val_df = add_image_data_df(val_df,hdf5_paths)
    test_df = add_image_data_df(test_df, hdf5_paths)

    # 5: convert to numpy arrays
    print('Clean Data Step 5')
    X_train = np.stack(train_df['img_arr'])
    X_val = np.stack(val_df['img_arr'])
    X_test = np.stack(test_df['img_arr'])

    # 6: Standardize the outputs: 
    y_train = np.array(train_df['bnpp_value_log'])
    y_val = np.array(val_df['bnpp_value_log'])
    y_test = np.array(test_df['bnpp_value_log'])

    # --- Standardize using TRAIN set statistics ---
    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std   
    y_test = (y_test - y_mean) / y_std

    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    print('XRayDataset objects created...')
    train_dataset, val_dataset, test_dataset = XRayDataset(X_train, y_train, transform), XRayDataset(X_val, y_val, transform), XRayDataset(X_test, y_test, transform)
    return train_dataset, val_dataset, test_dataset, y_mean, y_std

if __name__ == "__main__":
    main()
