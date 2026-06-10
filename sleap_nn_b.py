#!/usr/bin/env python3
"""Launch sleap-nn with TF32 matmul enabled on Ampere/Ada GPUs."""
import torch
torch.set_float32_matmul_precision('high')

from sleap_nn.cli import cli
cli()
