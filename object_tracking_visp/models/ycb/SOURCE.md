# YCB Mustard Bottle model

`006_mustard_bottle/google_16k` was downloaded from the official YCB dataset:

https://ycb-benchmarks.s3.amazonaws.com/data/google/006_mustard_bottle_google_16k.tgz

The dataset is licensed under CC BY 4.0. The YCB OBJ mesh is retained unchanged.
`dependencies/models/mustard.cao` is a compact convex hull derived from that mesh
and transformed to the DOPE Mustard frame and dimensions.

The official DOPE checkpoint is `Mustard.pth` from NVIDIA's public weights folder.
The ONNX file is exported from that checkpoint using the official DOPE network
definition in `dependencies/dope-source/models.py`. NVIDIA's Source Code License
is included at `dependencies/dope-source/LICENSE.md`; it limits use to
non-commercial research or evaluation.
