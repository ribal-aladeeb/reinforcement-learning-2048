import logging
import torch

if not torch.cuda.is_available():
    logging.warning("No GPU: Cuda is not utilized")
    device = "cpu"
else:
    device = "cuda:0"
