from Llama import TransformerBlockLlama
import torch
from utils import get_args, compare
import sys
import os
from kernels.gemv import gemv
import subprocess

if __name__ == "__main__":
  # GemV
  # gemv()

  # RMSNorm

  # partially functional only
  # generates traces for theh Dot Products on AiM
  # but uses a python-executed loop to aggregate the
  # sum of squares of input elements

  args = get_args()
  from kernels.dic_model import create_dic_model
  dic_model = create_dic_model(args)

  TB = TransformerBlockLlama(dic_model, args)
  TB.memory_mapping()
  TB.memory_mapping_verification()

  x_pow_sum = 0
  op_size = (TB.dic_shape["x_neighbor_bank"][0] - 1)