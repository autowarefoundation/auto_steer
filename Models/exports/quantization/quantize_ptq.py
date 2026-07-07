import os
import cv2
import sys
import copy
import tqdm
import torch
import warnings
import itertools

import numpy as np
import onnxruntime as ort

from pathlib import Path
from argparse import ArgumentParser

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from Models.data_utils.auto_steer.load_data_auto_steer import LoadDataAutoSteer
from Models.model_components.auto_steer.auto_steer_network import AutoSteerNetwork

from torchao.quantization.pt2e.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)

from torchao.quantization.pt2e import move_exported_model_to_eval, allow_exported_model_train_eval

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

warnings.filterwarnings(action='ignore', category=DeprecationWarning, module=r'.*')
warnings.filterwarnings(action='default', module=r'torchao.quantization.pt2e')

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)


# Evaluate functions
def run_onnx_model(session, image_tensor):
    # Get input name
    input_name = session.get_inputs()[0].name

    input_data = image_tensor.detach().cpu().numpy()
    input_data = np.expand_dims(input_data, axis=0)
    xp, h_vector = session.run(None, {input_name: input_data})
    xp = xp.squeeze()
    h_vector = h_vector.squeeze()

    return xp, h_vector


def compute_vector_ap(tp_dict, fp_dict, conf, target, eps=1e-16):
    """
    Computes AP for multiple thresholds, returns mean precision, mean recall,
    mAP@50 (threshold 0.05), and mean AP across all thresholds.
    """
    thresholds = sorted(tp_dict.keys())
    ap_dict = {}
    all_precisions = []
    all_recalls = []

    for t in thresholds:
        tp = np.array(tp_dict[t])
        fp = np.array(fp_dict[t])
        conf_t = np.array(conf)

        # Sort by confidence descending
        i = np.argsort(-conf_t)
        tp, fp = tp[i], fp[i]

        # Cumulative sums
        tpc = tp.cumsum()
        fpc = fp.cumsum()

        # Precision & Recall
        precision = tpc / (tpc + fpc + eps)
        recall = tpc / (len(target) + eps)

        # AP using 101-point interpolation
        m_rec = np.concatenate(([0.0], recall, [1.0]))
        m_pre = np.concatenate(([0.0], precision, [0.0]))
        m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))
        x = np.linspace(0, 1, 101)
        ap = np.trapezoid(np.interp(x, m_rec, m_pre), x)

        ap_dict[t] = ap
        all_precisions.append(precision.mean())
        all_recalls.append(recall.mean())

    # Mean precision & recall across thresholds
    m_pre_val = np.mean(all_precisions)
    m_rec_val = np.mean(all_recalls)

    # mAP@50 is the AP at threshold 0.05
    map50 = ap_dict[0.05]
    mean_ap = np.mean(list(ap_dict.values()))

    return m_pre_val, m_rec_val, mean_ap, map50


def evaluate(int8_model_path, dataset):
    # Create session with GPU support
    session = ort.InferenceSession(
        int8_model_path,
        providers=["CPUExecutionProvider"]
    )

    train_data_loader, val_dataloader = prepare_data_loaders(dataset)

    # Configure multiple normalized thresholds
    thresholds = [0.02, 0.05, 0.10]  # example thresholds
    all_tp = {t: [] for t in thresholds}
    all_fp = {t: [] for t in thresholds}
    all_conf, all_output, all_target = [], [], []

    p_bar = tqdm.tqdm(val_dataloader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP', 'mAP@50'))

    with torch.no_grad():
        for batch_idx, (samples, targets_xp, targets_h_vector) in enumerate(p_bar):
            samples = samples / 255.0

            batch_size = samples.shape[0]
            for i in range(batch_size):
                # Forward pass
                xp, h_vector = run_onnx_model(session, samples[i])

                target_xp = targets_xp[i].squeeze().cpu().numpy()
                target_h_vector = targets_h_vector[i].squeeze().cpu().numpy()

                # Apply mask (only valid lane points)
                h_vector_mask = (h_vector > 0.5)
                target_h_vector_mask = (target_h_vector > 0.5)

                line = (xp * h_vector_mask)
                target_line = (target_xp * target_h_vector_mask)

                # Distance in normalized space
                dist = np.abs(line - target_line).mean()
                conf = h_vector.mean()

                # Evaluate TP/FP for each threshold
                for t in thresholds:
                    if dist < t:
                        all_tp[t].append(1)
                        all_fp[t].append(0)
                    else:
                        all_tp[t].append(0)
                        all_fp[t].append(1)

                all_conf.append(conf)
                all_output.append(line)
                all_target.append(target_line)

        m_pre, m_rec, mean_ap, map50 = compute_vector_ap(all_tp, all_fp, all_conf, all_target)
        print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, mean_ap, map50))


# Visualization functions
def make_visualization(frame, xp, h_vector):
    yp = np.linspace(0, 511, 64, dtype=int)

    e_xp = xp * 1024
    e_h_vector = h_vector
    e_h_vector = (e_h_vector >= 0.5).astype(int)
    e_xp = e_xp * e_h_vector

    # Ego
    for x, y, h in zip(e_xp, yp, e_h_vector):
        if h == 1:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), thickness=-1)


def test_onnx_model(model_path, image_path):
    # Load model
    session = ort.InferenceSession(
        model_path,
        # providers=["CUDAExecutionProvider"]
        providers=["CPUExecutionProvider"]
    )

    # Get input name
    input_name = session.get_inputs()[0].name

    img = cv2.imread(image_path)
    image = cv2.imread(image_path)
    # Preprocess
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    # HWC → CHW
    image = np.transpose(image, (2, 0, 1))

    # Add batch dim
    image = np.expand_dims(image, axis=0)
    # Run inference
    xp, h_vector = session.run(None, {input_name: image})
    xp = xp.squeeze()
    h_vector = h_vector.squeeze()

    make_visualization(img, xp, h_vector)
    cv2.imshow('Prediction Objects', img)
    cv2.waitKey(0)  # IMPORTANT: wait until key press
    cv2.destroyAllWindows()


# Quantization functions
def load_model(model_path):
    model = AutoSteerNetwork().load_model(version='n', num_classes=4, checkpoint_path=model_path)
    return model


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")


def prepare_data_loaders(dataset, train_batch_size=30, val_batch_size=50):
    # train_batch_size = 30
    # val_batch_size = 50

    train_dir = Path(dataset + "/images/train/")
    filenames = [f.as_posix() for f in train_dir.rglob("*") if f.is_file()]
    train_dataset = LoadDataAutoSteer(filenames)

    val_dir = Path(dataset + "/images/val/")
    val_filenames = [f.as_posix() for f in val_dir.rglob("*") if f.is_file()]
    val_dataset = LoadDataAutoSteer(val_filenames)

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size,
        sampler=val_sampler)

    return train_dataloader, val_dataloader


def calibrate(model, data_loader, num_cal_samples=100):
    # num_cal_samples = 100
    move_exported_model_to_eval(model)
    iterator = list(itertools.islice(data_loader, num_cal_samples))
    pbar = tqdm.tqdm(iterator, total=num_cal_samples, desc="calibration")
    with torch.no_grad():
        # for samples, targets_xp, targets_h_vector in tqdm.tqdm(iterator, total=num_cal_samples, desc='calibration'):
        for samples, targets_xp, targets_h_vector in pbar:
            samples = samples / 255.0
            model(samples)


def quantize(dataset, fp32_model_path, int8_model_path, train_batch_size, val_batch_size, num_cal_samples):
    train_data_loader, val_dataloader = prepare_data_loaders(dataset)
    example_inputs = (torch.randn(50, 3, 512, 1024),)

    # Load model
    float_model = load_model(fp32_model_path).to("cpu")
    float_model.eval()

    # create another instance of the model since
    # we need to keep the original model around
    model_to_quantize = load_model(fp32_model_path).to("cpu")
    model_to_quantize.eval()

    # Export the model with torch.export

    # exported_model = torch.export.export(model_to_quantize, example_inputs).module()

    # for pytorch 2.6+
    dynamic_shapes = tuple(
        {0: torch.export.Dim("dim")} if i == 0 else None
        for i in range(len(example_inputs))
    )
    exported_model = torch.export.export(model_to_quantize, example_inputs, dynamic_shapes=dynamic_shapes).module()

    # Import the Backend Specific Quantizer and Configure how to Quantize the Model

    quantizer = XNNPACKQuantizer()
    qconfig = get_symmetric_quantization_config(is_per_channel=False, is_qat=False, )
    quantizer.set_global(qconfig)

    # Prepare the Model for PTQ and calibrate

    prepared_model = prepare_pt2e(exported_model, quantizer)
    # print(prepared_model.graph)

    move_exported_model_to_eval(prepared_model).cpu()
    calibrate(prepared_model, val_dataloader, num_cal_samples)  # run calibration on sample data

    # Export model to ONNX INT8 format, use batch size 1 for export
    input_tensor = torch.randn(1, 3, 512, 1024)
    example_inputs = (input_tensor,)

    int8_model = convert_pt2e(copy.deepcopy(prepared_model).cpu())
    move_exported_model_to_eval(int8_model)

    out_path = Path(args.int8_onnx_model_path)
    input_names = [f"input_{i}" for i in range(len(example_inputs))]
    output_names = ["output_xp", "output_h_vector"]
    model_cpu = int8_model.cpu()
    move_exported_model_to_eval(model_cpu)
    allow_exported_model_train_eval(model_cpu)
    model_cpu.eval()
    for module in model_cpu.modules():
        module.training = False
    torch.onnx.export(
        model_cpu, example_inputs, str(out_path),
        dynamo=True,
        input_names=input_names,
        output_names=output_names,
        external_data=False,
    )
    print(f"  ONNX saved ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = ArgumentParser(description="PTQ quantization for AutoSteer 2.0")

    parser.add_argument('-q', '--quantize', action='store_true', required=False, help="Quantize model")
    parser.add_argument('-e', '--evaluate', action='store_true', required=False, help="Evaluate model")
    parser.add_argument('-t', '--test', action='store_true', required=False,
                        help="Test model inference and visualize results")
    parser.add_argument('-d', '--dataset', help="dataset directory path")
    parser.add_argument('-tbs', '--train_batch_size', help="Dataset test batch size")
    parser.add_argument('-vbs', '--val_batch_size', help="Dataset validation batch size")
    parser.add_argument('-ncs', '--num_cal_samples', help="Number of calibration batch samples")
    parser.add_argument("-fp32", "--fp32_model_path", dest="fp32_model_path", default="best_model.pth",
                        help="FP32 model path")
    parser.add_argument("-int8", "--int8_model_path", dest="int8_model_path", default="autosteer_2.0_int8.pt",
                        help="INT8 model path")
    parser.add_argument("-int8onnx", "--int8_onnx_model_path", dest="int8_onnx_model_path",
                        default="autosteer_2.0_INT8.pt",
                        help="INT8 model path")
    parser.add_argument('-i', '--input', help="Input test image")

    args = parser.parse_args()
    dataset = args.dataset

    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    num_cal_samples = args.num_cal_samples

    fp32_model_path = args.fp32_model_path
    int8_model_path = args.int8_model_path
    int8_onnx_model_path = args.int8_onnx_model_path

    if args.quantize:
        quantize(dataset, fp32_model_path, int8_model_path, train_batch_size, val_batch_size, num_cal_samples)
    elif args.evaluate:
        evaluate(int8_onnx_model_path, dataset)
    elif args.test:
        test_onnx_model(args.int8_onnx_model_path, args.input)
    else:
        print("No quantization, evaluation or test parameters provided!")
