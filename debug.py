import torch
import ipdb

indexer_weight = torch.load("/aifs4su/guhao/checkpoints/qwen-1.5b-indexer_score/pytorch_model.bin", map_location="cpu")
indexer_keys = [k for k in indexer_weight.keys() if "indexer" in k]
print(indexer_keys)
# ipdb.set_trace()