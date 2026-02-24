import argparse
from src.generate_data import main
from src.logger import get_logger
logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic healthcare data.")
    parser.add_argument("--n-samples", type=int,
                        help="Number of samples to generate")
    parser.add_argument("--seed", type=int,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str,
                        help="Path to save the generated file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.n_samples or args.seed or args.output:
        logger.info(f"Wrapper: Arguments received.")
        logger.info(f"  Samples: {args.n_samples}")
        logger.info(f"  Seed: {args.seed}")
        logger.info(f"  Output: {args.output}")
        logger.info("Note: In this version, configuration is primarily driven by src/config.py.")
        logger.info(
            "      To fully override, update src/config.py or extend src/generate_data.py")

    main()

