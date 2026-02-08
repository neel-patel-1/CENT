import torch
from utils import get_args

def create_dic_model(args=None):
  if args is None:
    args = get_args()
  head_dim = 128
  dim = head_dim * args.n_heads
  ffn_dim = args.ffn_dim
  TP_param = 8 if args.GPT3_175B_TP_8 else 1
  n_heads = args.n_heads // TP_param
  n_kv_heads = args.n_kv_heads if args.Llama_GQA else n_heads
  dic_model = {
      "TP_param": torch.tensor(TP_param),
      "dim": torch.tensor(dim),
      "n_heads": torch.tensor(n_heads),
      "x": torch.zeros((1, 1, dim)),
      "SANorm": torch.zeros(dim),
      "FFNNorm": torch.zeros(dim),
      "sa": torch.zeros((1, 1, dim)),
      "h": torch.zeros((1, 1, dim)),
      "out": torch.zeros((1, 1, dim)),
      "wq": torch.zeros((dim // TP_param, dim)),
      "wk": torch.zeros((head_dim * n_kv_heads, dim)),
      "wv": torch.zeros((head_dim * n_kv_heads, dim)),
      "xq": torch.zeros((1, 1, dim)),
      "xk": torch.zeros((1, 1, head_dim * n_heads)),
      "xv": torch.zeros((1, 1, head_dim * n_heads)),
      "start_pos": torch.tensor(args.seqlen - 1),
      "cache_k": torch.zeros((1, args.seqlen, n_kv_heads, head_dim)),
      "cache_v": torch.zeros((1, args.seqlen, n_kv_heads, head_dim)),
      "scores": torch.zeros((1, n_heads, 1, args.seqlen)),
      "output": torch.zeros((1, 1, dim)),
      "wo": torch.zeros((dim // TP_param, dim)),
      "w1": torch.zeros((ffn_dim // TP_param, dim)),
      "w3": torch.zeros((ffn_dim // TP_param, dim)),
      "w2": torch.zeros((dim // TP_param, ffn_dim)),
      "ffn": torch.zeros((1, 1, dim))
  }
  if args.Llama_GQA:
    dic_model["n_kv_heads"] = torch.tensor(n_kv_heads)

  return dic_model