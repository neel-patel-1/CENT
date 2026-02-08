from Llama import TransformerBlockLlama
from model_config import create_dic_model
import os
from utils import get_args

def RMSNorm():
  args = get_args()
  dic_model = create_dic_model(args)

  kernel_name = "RMSNorm_{D}"
  outdir = f"./traces/"
  os.makedirs(outdir, exist_ok=True)
  # Create input vector
  D=512