# %%
# Comment above is for Jupyter execution in VSCode
# ! /usr/bin/env python3
import sys
import math
import torch
from PIL import Image
from torchvision import transforms

import numpy as np

sys.path.append('..')
from Models.model_components.auto_steer.auto_steer_network import AutoSteerNetwork


class AutoSteerNetworkInfer():
    def __init__(self, autosteer_checkpoint_path=''):
        # Image loader
        self.image_loader = transforms.Compose([
            transforms.Lambda(lambda img: img[:, :, ::-1].copy()),  # BGR → RGB
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')

        # Instantiate model, load to device and set to evaluation mode
        if (len(autosteer_checkpoint_path) > 0):
            # Loading model with full pre-trained weights
            self.model = AutoSteerNetwork().load_model(version='n', num_classes=4, checkpoint_path=autosteer_checkpoint_path)
        else:
            raise ValueError('No path to checkpiont file provided in class initialization')

        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    @torch.no_grad()
    def inference(self, image):
        H, W = image.shape[:2]
        if (W != 1024 or H != 512):
            raise ValueError('Incorrect input size - input image must have height of 1024px and width of 512px')

        image_tensor = self.image_loader(image).unsqueeze(0).to(self.device)

        # Run model
        xp, h_vector = self.model(image_tensor)
        xp = xp.detach().cpu().numpy()
        xp = xp.squeeze()  # (2, 64)
        h_vector = h_vector.detach().cpu().numpy()
        h_vector = h_vector.squeeze()  # (2,64)

        return  xp, h_vector
