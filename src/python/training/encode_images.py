
from pathlib import Path

import torch

from functions.networks.nets import VQVAE
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tqdm import tqdm
from util import get_test_dataloader
import nibabel as nib

set_determinism(seed=42)
print_config()


all_latents = []

# Define range of epochs and ROI sizes
epoch = 320
roi_sizes = [[512,512,256]]  # Example of varying ROI sizes

output_dir = Path(f"./output")
output_dir.mkdir(exist_ok=True, parents=True)

for roi_size in roi_sizes:
    test_loader = get_test_dataloader(
        batch_size=1,
        cache_rate=1.0,
        roi_size=roi_size,
        test_ids="./validation.csv",
        num_workers=4,
    )
    

    config_path = "./vqgan_ds4.yaml"
    stage1_ckpt_path = f"./checkpoint_epoch_{epoch}.pth"

    print(f"Getting data for epoch {epoch}, roi_size {roi_size}...")

    print("Creating model...")
    device = torch.device("cuda")
    config = OmegaConf.load(config_path)
    stage1 = VQVAE(**config["stage1"]["params"])
    checkpoint = torch.load(stage1_ckpt_path)

    stage1.load_state_dict(checkpoint["state_dict"])
    stage1 = stage1.to(device)
    stage1.eval()

    print(f"Encoding latents for epoch {epoch}, roi_size {roi_size}...")
    for batch in tqdm(test_loader):
        x = batch["image"].to(device)
        print(x.shape)
        new_affine = batch["image"].meta["affine"].numpy()

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=True):
                latents = stage1.encode_stage_2_inputs(x)
                


        full_path = batch["image"].meta["filename_or_obj"]
        filename = Path(full_path).stem  # Extract filename without extension

        # Save the latent as a .nii.gz file
        latent_path = output_dir / f"{filename}_ds4.nii.gz"
        latent_image = nib.Nifti1Image(latents.cpu().numpy()[0], affine=new_affine)  # Assuming 3D latents
        nib.save(latent_image, latent_path)

print("Latents saved successfully.")