import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args():
    parser = argparse.ArgumentParser()

    model = parser.add_subparsers(dest='model', required=True)
    dinov2 = model.add_parser("DINOV2")
    dinov2.add_argument("--checkpoint", type=Path, required=False, help="path to checkpoint")

    dinov3 = model.add_parser("DINOV3")
    dinov3.add_argument("--checkpoint", type=Path, required=False, help="path to checkpoint")

    sam = model.add_parser("SAM")
    sam.add_argument("--checkpoint", type=Path, required=False, help="path to checkpoint")

    return parser.parse_args()

def main():
    args = parse_args()

if __name__ == '__main__':
    main()