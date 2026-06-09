from pathlib import Path


HF_DATASET_ID = "zhangdw/GABench"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def benchmark_csv_path(root: Path | None = None) -> Path:
    base = root or project_root()
    return base / "benchmark" / "benchmark.csv"


def dataset_dir(root: Path | None = None) -> Path:
    base = root or project_root()
    return base / "dataset"


def dataset_result_dir(root: Path | None = None) -> Path:
    return dataset_dir(root) / "result"


def download_command(target_dir: str = ".") -> str:
    return (
        f"uvx --from huggingface_hub hf download {HF_DATASET_ID} "
        f"--repo-type dataset --local-dir {target_dir} "
        "--include 'benchmark/**' --include 'dataset/**'"
    )


def missing_data_message(path: Path) -> str:
    return (
        f"Required benchmark data is missing: {path}\n"
        "Download the migrated data from Hugging Face with:\n"
        f"  {download_command()}"
    )
