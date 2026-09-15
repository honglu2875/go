#!/usr/bin/env python3
"""SPMD entrypoint; retain the full parent learner as train_student.py."""
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', required=True)
    args, _ = parser.parse_known_args()
    with open(args.config) as stream:
        configuration = json.load(stream)
    if configuration.get('kind') == 'student_calibration':
        from diagnose import main
    else:
        from train_student import main
    main()
