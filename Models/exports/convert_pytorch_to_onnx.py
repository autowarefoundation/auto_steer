#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import onnx
from argparse import ArgumentParser
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from Models.model_components.auto_steer.auto_steer_network import AutoSteerNetwork

def main():

    # Argument parser for data root path and save path
    parser = ArgumentParser()

    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", required=True, \
                        help="path to pytorch checkpoint file to load model dict")

    parser.add_argument("-o", "--onnx_model_path", dest="onnx_model_path", required=True, \
                        help="path to converted ONNX model, must include output file name with .onnx extension")

    args = parser.parse_args()

    # Get input arguments
    model_checkpoint_path = args.model_checkpoint_path
    onnx_model_path = args.onnx_model_path

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference')


    # Instantiating Model and setting to evaluation mode
    model = AutoSteerNetwork().build_model(version='n')

    # Loading Pytorch checkpoint
    print('Loading Network')
    if(len(model_checkpoint_path) > 0):
            checkpoint = torch.load(model_checkpoint_path, weights_only=False, map_location=device)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                # If checkpoint['model'] is a model object, extract its state_dict
                if hasattr(checkpoint['model'], 'state_dict'):
                    model.load_state_dict(checkpoint['model'].state_dict())
                else:
                    model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
    else:
        raise ValueError('No path to checkpiont file provided in class initialization')
    model = model.to(device)
    model = model.eval()

    # Fake input data

    input_shape=(1, 3, 512, 1024)
    input_data = torch.randn(input_shape).to(device)

    # Export FP32 model to onnx
    print('Converting model to ONNX at FP32 and exporting')

    # AutoSteer 2.0 returns a 2-tuple (lane_value, height)
    torch.onnx.export(model,
                    input_data,
                    onnx_model_path,
                    export_params=True,
                    opset_version=18,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['lane_value', 'height'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'lane_value': {0: 'batch_size'},
                        'height': {0: 'batch_size'},
                    },
                    external_data=False)

    # Run checks on exported FP32 ONNX network
    ONNX_network = onnx.load(onnx_model_path)
    onnx.checker.check_model(ONNX_network)
    print('Checks passed - export complete')

if __name__ == '__main__':
  main()
# %%