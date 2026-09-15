"""Deterministic CPU probe for recipe packaging; this does not train a model."""

import argparse
import json
from pathlib import Path
import random

from gozero.snapshots import canonical_json, digest, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if config.get("schema_version") != 1 or set(config) != {"schema_version", "seed", "samples"}:
        raise ValueError("Expected schema_version, seed, and samples")
    if type(config["seed"]) is not int or type(config["samples"]) is not int or config["samples"] <= 0:
        raise ValueError("seed must be an integer and samples must be a positive integer")
    rng = random.Random(config["seed"])
    payload = [rng.getrandbits(32) for _ in range(config["samples"])]
    result = {"schema_version": 1, "kind": "harness_probe", "config_sha256": digest(canonical_json(config)),
              "payload_sha256": digest(canonical_json(payload)), "samples": len(payload)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "result.json").write_bytes(canonical_json(result))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
