from Llama import TransformerBlockLlama
import torch
from utils import get_args, compare
from kernels.dic_model import create_dic_model
import sys
import os
import subprocess

def gemv():
  args = get_args()
  dic_model = create_dic_model(args)

  kernel_name = "gemv_{M}x{K}x{N}"
  outdir = f"./traces/"
  os.makedirs(outdir, exist_ok=True)
  # Create matrix and vector
  D=512

  M=D
  K=D
  N=D

  vector = torch.arange(M, dtype=torch.float16)                     # shape (16,)
  matrix = torch.arange(K*N, dtype=torch.float16).reshape(K, N)  # shape (16,16)

  # create a TransformerBlock which provides AiM Instruction Generation Functions
  TB = TransformerBlockLlama(dic_model, args)
  TB.trace_file = f'{outdir}/{kernel_name.format(M=M,K=K,N=N)}.trace'
  TB.file = open(TB.trace_file, 'w')

  TB.memory_mapping()
  # TB.memory_mapping_verification()

  row_idx = getattr(TB, "wq_row_index", 0)
  channel_lst = [0]
  total_banks = getattr(TB, "FC_total_banks", TB.total_banks if hasattr(TB, "total_banks") else 1)

  ref = torch.matmul(vector.float(), matrix.float())   # shape (N,)
  pim_out = TB.Vector_Matrix_Mul_weight_pim(vector, row_idx, M, N, total_banks, True, "breakdown_ffn_weight")
  TB.finish()
  TB.file.close()

  compare(pim_out, ref, "GEMV verification")

  command=f"../aim_simulator/build/ramulator2 -f ../aim_simulator/test/example.yaml -t {TB.trace_file}"
  log_file = f"./traces/{kernel_name.format(M=M,K=K,N=N)}.log"
  with open(log_file, "w") as lf:
    subprocess.run(command, shell=True, stdout=lf, stderr=lf, text=True)