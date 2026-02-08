from Llama import TransformerBlockLlama
import torch
from utils import get_args, compare
import sys
import os
from kernels.gemv import gemv
import subprocess

if __name__ == "__main__":
  # GemV
  gemv()
