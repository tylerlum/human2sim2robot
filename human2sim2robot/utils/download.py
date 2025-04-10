import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
import tyro
from tqdm import tqdm

from human2sim2robot.sim_training import get_repo_root_dir

TASK_NAMES = ["snackbox_push", "plate_pivotliftrack"]


@dataclass
class DownloadArgs:
    download_url: str

    # Assets
    include_assets: bool = False

    # Data
    task_name: Optional[str] = None
    all_tasks: bool = False
    include_human_demo_raw_data: bool = False
    include_human_demo_processed_data: bool = False

    # Models
    include_pretrained_models: bool = False

    @property
    def download_url_no_trailing_slash(self) -> str:
        if self.download_url.endswith("/"):
            return self.download_url[:-1]
        return self.download_url


def download_and_extract_zip(url: str, extract_to: Path) -> None:
    assert url.endswith(".zip"), f"URL must end with .zip, got {url}"

    url_filename_without_ext = Path(urlparse(url).path).stem
    output_zip_path = extract_to / f"{url_filename_without_ext}.zip"
    output_folder = extract_to / url_filename_without_ext
    print("=" * 80)
    print("Planning to:")
    print(f"Download {url} => {output_zip_path}")
    print(f"Then extract {output_zip_path} => {extract_to}")
    print(f"Then expect to end with {output_folder}")
    print("=" * 80 + "\n")

    if output_folder.exists():
        print("!" * 80)
        print(f"Folder {output_folder} already exists, skipping download.")
        print("!" * 80 + "\n")
        return

    # Make the directory
    extract_to.mkdir(parents=True, exist_ok=True)

    # Stream the download and show progress
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    # fmt: off
    with output_zip_path.open("wb") as file, tqdm(
        desc=f"Downloading {url}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    ) as bar:
    # fmt: on
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    # Extract the zip file with progress bar
    with zipfile.ZipFile(output_zip_path, "r") as zip_ref:
        zip_info_list = zip_ref.infolist()
        total_files = len(zip_info_list)

        with tqdm(
            desc=f"Extracting {output_zip_path}",
            total=total_files,
            unit="file",
            dynamic_ncols=True,
        ) as bar:
            for zip_info in zip_info_list:
                zip_ref.extract(zip_info, extract_to)
                bar.update(1)

    assert output_folder.exists(), f"Expected {output_folder} to exist"

    # Clean up the downloaded zip file
    print(f"Removing {output_zip_path}")
    output_zip_path.unlink()

    print("DONE!")
    print("~" * 80 + "\n")

    # Sleep between downloads to avoid request errors
    SLEEP_TIME_SECONDS = 1
    print(f"Sleeping for {SLEEP_TIME_SECONDS} seconds...")
    time.sleep(SLEEP_TIME_SECONDS)


def run_download(args: DownloadArgs) -> None:
    root_dir = get_repo_root_dir()

    # Assets
    if args.include_assets:
        url = f"{args.download_url_no_trailing_slash}/assets.zip"
        extract_to = root_dir
        download_and_extract_zip(url=url, extract_to=extract_to)

    # Raw data
    if args.include_human_demo_raw_data:
        if args.all_tasks:
            for task_name in TASK_NAMES:
                url = f"{args.download_url_no_trailing_slash}/data/human_demo_raw_data/{task_name}.zip"
                extract_to = root_dir / "data" / "human_demo_raw_data"
                download_and_extract_zip(url=url, extract_to=extract_to)
        else:
            assert args.task_name is not None and args.task_name in TASK_NAMES, (
                f"task_name must be one of {TASK_NAMES}, got {args.task_name}"
            )
            url = f"{args.download_url_no_trailing_slash}/data/human_demo_raw_data/{args.task_name}.zip"
            extract_to = root_dir / "data" / "human_demo_raw_data"
            download_and_extract_zip(url=url, extract_to=extract_to)

    # Processed data
    if args.include_human_demo_processed_data:
        if args.all_tasks:
            for task_name in TASK_NAMES:
                url = f"{args.download_url_no_trailing_slash}/data/human_demo_processed_data/{task_name}.zip"
                extract_to = root_dir / "data" / "human_demo_processed_data"
                download_and_extract_zip(url=url, extract_to=extract_to)
        else:
            assert args.task_name is not None and args.task_name in TASK_NAMES, (
                f"task_name must be one of {TASK_NAMES}, got {args.task_name}"
            )
            url = f"{args.download_url_no_trailing_slash}/data/human_demo_processed_data/{args.task_name}.zip"
            extract_to = root_dir / "data" / "human_demo_processed_data"
            download_and_extract_zip(url=url, extract_to=extract_to)

    # Pretrained models
    if args.include_pretrained_models:
        url = f"{args.download_url_no_trailing_slash}/pretrained_models.zip"
        extract_to = root_dir
        download_and_extract_zip(url=url, extract_to=extract_to)


def main() -> None:
    args = tyro.cli(DownloadArgs)
    run_download(args)


if __name__ == "__main__":
    main()
