"""Create the raw routing input without fitting any text features."""
from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=Path(__file__).parent / "train.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "arena_slim.csv")
    args = parser.parse_args()
    columns = ["id", "model_a", "model_b", "prompt", "winner_model_a", "winner_model_b", "winner_tie"]
    frame = pd.read_csv(args.input, usecols=columns)
    if frame.id.duplicated().any():
        raise ValueError("Battle IDs must be unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} raw battles to {args.output}; no feature transform was fitted.")


if __name__ == "__main__":
    main()
