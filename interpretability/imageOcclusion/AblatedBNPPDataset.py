import os
import h5py
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


class AblatedBNPPDataset(Dataset):
    def __init__(self, predictions_df, fill_mode, patch_coords=None, ablate=True):
        """
        predictions_df: DataFrame with index = image IDs
        patch_coords: (r1, r2, c1, c2) in 256x256 space
        ablate: whether to apply ablation
        """

        home_dir = os.path.expanduser("~")
        self.base = os.path.join(home_dir, "teams", "b1")
        self.df = predictions_df.copy()
        self.df.index = self.df.index.astype(str)
        self.patch_coords = patch_coords
        self.ablate = ablate
        self.fill_mode = fill_mode
        self.normalize = transforms.Normalize(mean=[0.485], std=[0.229])
        # Build shard lookup
        self.key_to_file = self._build_key_to_file_map()
        self.keys = [k for k in self.df.index if k in self.key_to_file]

        print(f"Loaded {len(self.keys)} test images across shards.")

    def _preprocess_image(self, img_1024, target_size=256):
        H, W = img_1024.shape
        assert H == W

        factor = H // target_size

        # 1024 -> 256 via 4x4 block averaging
        img = img_1024.reshape(
            target_size, factor,
            target_size, factor
        ).mean(axis=(1, 3))

        img = img.astype(np.float32)

        # Per-image min-max normalization
        min_val = img.min()
        max_val = img.max()

        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)

        # Convert to tensor (1, 256, 256)
        img = torch.from_numpy(img).float().unsqueeze(0)

        # Normalize to training distribution
        img = self.normalize(img)

        return img

    # --------------------------------------------------
    # APPLY ABLATION
    # --------------------------------------------------
    def _apply_ablation(self, img_tensor):
        if not self.ablate:
            return img_tensor
    
        if self.patch_coords is None:
            raise ValueError("patch_coords must be provided if ablate=True")
    
        r1, r2, c1, c2 = self.patch_coords
    
        assert 0 <= r1 < r2 <= 256
        assert 0 <= c1 < c2 <= 256
    
        img_tensor = img_tensor.clone()
    
        if self.fill_mode == "mean":
            # 0 in normalized space = training mean
            img_tensor[:, r1:r2, c1:c2] = 0.0
    
        elif self.fill_mode == "noise":
            patch = img_tensor[:, r1:r2, c1:c2]
            noise = torch.randn_like(patch) * patch.std() + patch.mean()
            img_tensor[:, r1:r2, c1:c2] = noise
    
        elif self.fill_mode == "blur":
            # simple average pooling blur on region
            patch = img_tensor[:, r1:r2, c1:c2].unsqueeze(0)  # add batch dim
            blurred = torch.nn.functional.avg_pool2d(
                patch,
                kernel_size=7,
                stride=1,
                padding=3
            )
            img_tensor[:, r1:r2, c1:c2] = blurred.squeeze(0)
    
        elif self.fill_mode == "zero_raw":
            # corresponds to raw pixel 0 after normalization
            raw_zero = (0.0 - 0.485) / 0.229
            img_tensor[:, r1:r2, c1:c2] = raw_zero
    
        else:
            raise ValueError(f"Unknown fill_mode: {fill_mode}")
    
        return img_tensor
    
    def _build_key_to_file_map(self):

        hdf5_names = [f'bnpp_frontalonly_1024_{i}' for i in range(1, 11)]
        hdf5_names.append('bnpp_frontalonly_1024_0_1')

        test_keys = set(self.df.index)
        key_to_file = {}

        for name in hdf5_names:
            path = os.path.join(self.base, f"{name}.hdf5")
            if not os.path.exists(path):
                continue

            with h5py.File(path, "r") as f:
                file_keys = set(f.keys())
                matches = test_keys.intersection(file_keys)

                for key in matches:
                    key_to_file[key] = name

        return key_to_file
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        image_id = self.keys[idx]
        hdf5_file = self.key_to_file[image_id]

        path = os.path.join(self.base, hdf5_file + ".hdf5")

        with h5py.File(path, "r") as f:
            img_1024 = f[image_id][()]

        img_tensor = self._preprocess_image(img_1024)
        img_tensor = self._apply_ablation(img_tensor)

        assert img_tensor.shape == (1, 256, 256)

        return img_tensor, image_id
