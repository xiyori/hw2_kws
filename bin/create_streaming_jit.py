import torch
import argparse
import sys
import os

sys.path += [os.getcwd()]

from src.models import StreamingCRNN


parser = argparse.ArgumentParser(description="Save streaming KWS model in TorchScript format.")
parser.add_argument("path", nargs="?", metavar="PATH", type=str,
                    default="resources/checkpoints/crnn_epoch19.pth",
                    help="Path to offline KWS model save (default: %(default)s).")
args = parser.parse_args()

streaming_kws = torch.jit.script(StreamingCRNN(args.path))
streaming_kws.save("resources/checkpoints/streaming_kws.pth")
