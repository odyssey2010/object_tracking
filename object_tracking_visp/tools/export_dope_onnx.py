import argparse
import os
import sys
from pathlib import Path

import onnx
import torch
from torch import nn


class BeliefMapExport(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, image):
        belief_maps, _ = self.network(image)
        return belief_maps[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Export the official DOPE Mustard checkpoint to ONNX."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source_root = Path(__file__).resolve().parents[1] / "dependencies" / "dope-source"
    sys.path.insert(0, str(source_root))
    from models import DopeNetwork

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected the checkpoint to contain a state dictionary.")

    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }
    network = DopeNetwork(pretrained=False, numBeliefMap=9, numAffinity=16)
    network.load_state_dict(state_dict, strict=True)
    network.eval()

    sample = torch.zeros(1, 3, 448, 448, dtype=torch.float32)
    with torch.no_grad():
        belief_maps = BeliefMapExport(network)(sample)
    if tuple(belief_maps.shape) != (1, 9, 56, 56):
        raise RuntimeError("Unexpected DOPE output shape: {}".format(tuple(belief_maps.shape)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        torch.onnx.export(
            BeliefMapExport(network),
            sample,
            str(temporary_output),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["belief_maps"],
        )
        onnx.checker.check_model(onnx.load(str(temporary_output)))
        os.replace(str(temporary_output), str(args.output))
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    print("Exported {} ({})".format(args.output, belief_maps.shape))


if __name__ == "__main__":
    main()
