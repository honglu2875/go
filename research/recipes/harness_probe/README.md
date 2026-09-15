This is a cloneable deterministic CPU probe for source packaging. Its `train.py` writes a seed-dependent payload hash and configuration hash. It contains no model, Go environment, or claimed performance measurement. It is intentionally independent of JAX so the snapshot foundation can be validated before TPU runtime qualification.

Use it to test clone/freeze/restore behavior. Scientific recipes will own their complete plain-JAX model and training implementation, following the supplied rig example, and use the shared library for common infrastructure.
