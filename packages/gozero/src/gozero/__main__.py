"""Small source-management CLI; accelerator launch is a separate milestone."""

import argparse
from pathlib import Path
import sys

from .snapshots import clone_recipe, freeze, verify


def main() -> int:
    parser = argparse.ArgumentParser(prog="gozero")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    clone = commands.add_parser("clone", help="Clone a complete editable research recipe")
    clone.add_argument("recipe", type=Path)
    clone.add_argument("name")
    snapshot = commands.add_parser("snapshot", help="Freeze source, dependency locks, and resolved JSON config")
    snapshot.add_argument("recipe", type=Path)
    snapshot.add_argument("--config", type=Path, required=True)
    snapshot.add_argument("--store", type=Path)
    check = commands.add_parser("verify", help="Verify all snapshot contents and reject added files")
    check.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "clone":
            print(clone_recipe(args.repo, args.recipe, args.name))
        elif args.command == "snapshot":
            store = args.store or args.repo / ".gozero" / "snapshots"
            config = args.config if args.config.is_absolute() else args.repo / args.config
            print(freeze(args.repo, args.recipe, config, store))
        else:
            print(verify(args.snapshot)["snapshot_id"])
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"gozero: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
