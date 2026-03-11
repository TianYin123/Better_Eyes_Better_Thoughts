import argparse
import os
import subprocess
import sys

DEFAULT_MODELS = [
    "<MODEL_B>",
]

DATASET_LIST = [
    "PMC_VQA",
    "VQA_RAD",
]


def parse_bool(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch launcher for base_evaluate.py across models/datasets/reasoning modes using the full dataset."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Test model list")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional subset of dataset names from DATASET_LIST to run",
    )
    parser.add_argument("--reasoning-modes", nargs="+", default=["true", "false"], help="Reasoning flags: true/false")
    parser.add_argument("--judge-model-name", default="<JUDGE_MODEL_NAME>")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay-base", type=float, default=5.0)
    parser.add_argument(
        "--script-path",
        default=os.path.join(os.path.dirname(__file__), "base_evaluate.py"),
        help="Path to base_evaluate.py",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not DATASET_LIST:
        raise ValueError("DATASET_LIST is empty. Please edit run_batch_exp7.py and add datasets.")

    reasoning_modes = [parse_bool(value) for value in args.reasoning_modes]
    if args.datasets is None:
        dataset_names = list(DATASET_LIST)
    else:
        dataset_names = []
        for dataset_name in args.datasets:
            if dataset_name not in DATASET_LIST:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found in DATASET_LIST. "
                    f"Available keys: {list(DATASET_LIST)}"
                )
            dataset_names.append(dataset_name)

    total_jobs = len(args.models) * len(dataset_names) * len(reasoning_modes)
    job_index = 0
    failed_jobs = []

    print(f"Total jobs: {total_jobs}")

    for dataset_name in dataset_names:
        for model_name in args.models:
            for is_reasoning in reasoning_modes:
                job_index += 1
                print(
                    f"\n[{job_index}/{total_jobs}] dataset={dataset_name}, "
                    f"model={model_name}, reasoning={is_reasoning}"
                )

                cmd = [
                    sys.executable,
                    args.script_path,
                    "--dataset-name",
                    dataset_name,
                    "--test-model-name",
                    model_name,
                    "--judge-model-name",
                    args.judge_model_name,
                    "--is-reasoning",
                    str(is_reasoning).lower(),
                    "--max-workers",
                    str(args.max_workers),
                    "--max-retries",
                    str(args.max_retries),
                    "--retry-delay-base",
                    str(args.retry_delay_base),
                ]

                completed = subprocess.run(cmd, check=False)
                if completed.returncode != 0:
                    failed_jobs.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "reasoning": is_reasoning,
                            "returncode": completed.returncode,
                        }
                    )

    if failed_jobs:
        print("\nFailed jobs:")
        for item in failed_jobs:
            print(item)
        raise SystemExit(1)

    print("\nAll jobs finished successfully.")


if __name__ == "__main__":
    main()
