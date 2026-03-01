U-Net Toy Segmentation (PyTorch)
================================

**Overview**
- **Project**: Minimal U-Net implementation and a synthetic circle dataset for quick experimentation with segmentation in PyTorch.
- **Purpose**: Provide a compact example to train and inspect a U-Net on simple binary masks.

**Requirements**
- **Python**: 3.8+
- **Packages**: `torch`, `torchvision` (optional), `matplotlib`, `numpy`

Quick install (CPU):
```bash
python -m venv .venv
# Windows (cmd)
.venv\Scripts\activate
pip install --upgrade pip
pip install torch matplotlib numpy
```

If you have CUDA, install a matching `torch` build from https://pytorch.org.

**Files**
- [dataset.py](dataset.py): Synthetic `ShapeDataset` that creates circle images and masks. Run `python dataset.py` to visualize a sample.
- [u_net.py](u_net.py): `UNet` model and a small smoke-test in `if __name__ == '__main__'`.
- [train.py](train.py): Training loop that saves `unet_small.pth` after training.

**Quickstart**
- Visualize a dataset sample: `python dataset.py` (opens a matplotlib window showing image & mask).
- Train a small model: `python train.py` — saves `unet_small.pth` in the working directory.

**Training details**
- Model: `UNet(in_channels=1, out_channels=1)` in `u_net.py`.
- Loss: `BCEWithLogitsLoss` (outputs are raw logits; apply `torch.sigmoid` when evaluating probabilities).
- Checkpoint: `unet_small.pth` saved by `train.py`.

**Notes & Tips**
- Input / output shapes use single-channel grayscale images (shape: `N x 1 x H x W`).
- To evaluate predictions visually after training, load the model weights and run `torch.sigmoid(model(img))`.
- For reproducible experiments, seed NumPy and torch RNGs in your scripts.

**Next steps I can help with**
- Create a `requirements.txt` or `pyproject.toml`.
- Add an evaluation/visualization script that loads `unet_small.pth` and shows predictions.
- Add command-line args to `train.py` for hyperparameters.

**License**
- MIT
