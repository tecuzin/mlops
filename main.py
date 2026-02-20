import logging
import sys

from src.pipeline import mlops_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/finetune_rag_qa.yaml"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    mlops_pipeline(config_path=config_path, output_dir=output_dir)


if __name__ == "__main__":
    main()
