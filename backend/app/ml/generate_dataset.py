import argparse

from app.ml.synthetic_generator import SyntheticDataConfig, SyntheticWorkloadGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ML cluster workload data.")
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, default="data/synthetic_jobs.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SyntheticDataConfig(
        num_samples=args.num_samples,
        seed=args.seed,
        output_path=args.output_path,
    )
    generator = SyntheticWorkloadGenerator(config)
    frame = generator.generate()
    output = generator.export_csv(frame)
    print(f"Generated {len(frame)} samples to {output}")


if __name__ == "__main__":
    main()

