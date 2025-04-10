import subprocess
from dataclasses import dataclass
from pathlib import Path

import tyro
from tqdm import tqdm

from human2sim2robot.sim_training import get_package_root_dir, get_repo_root_dir


@dataclass
class SetupZshTabCompletionArgs:
    overwrite: bool = False
    catch_exceptions: bool = True
    output_folder: Path = get_repo_root_dir() / ".zsh_tab_completion"


def main() -> None:
    args = tyro.cli(SetupZshTabCompletionArgs)

    # Get tyro scripts
    print("Getting tyro scripts...")
    package_folder = get_package_root_dir()
    result = subprocess.run(
        f"grep -rl 'tyro.cli(' {package_folder}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if not result.stdout:
        print("No tyro scripts found.")
        exit()

    script_paths = [Path(x) for x in result.stdout.strip().split("\n")]
    print(f"Found {len(script_paths)} scripts in the package folder {package_folder}")
    for script_path in script_paths:
        assert script_path.exists(), f"{script_path} does not exist"

    # https://brentyi.github.io/tyro/tab_completion/#tab-completion
    args.output_folder.mkdir(exist_ok=True)
    print(f"Writing tab completion scripts to {args.output_folder}")

    for script_path in tqdm(
        script_paths, desc="Writing tab completion scripts", dynamic_ncols=True
    ):
        # Skip if output exists
        relative_script_path = script_path.relative_to(package_folder)
        output_path = (
            args.output_folder
            / f"_{str(relative_script_path).replace('.py', '_completion_py').replace('/', '__')}"
        )
        if output_path.exists() and not args.overwrite:
            print(f"Skipping {script_path} because {output_path} already exists")
            continue

        # Write completion script
        print(f"{script_path} -> {output_path} ...")
        completion_script = (
            f"python {script_path} --tyro-write-completion zsh {output_path}"
        )
        try:
            subprocess.run(completion_script, shell=True, check=True)
        except Exception as e:
            if not args.catch_exceptions:
                raise e
            print("!" * 80)
            print(f"Exception: {e}")
            print(f"Skipping {script_path} and continuing")
            print("!" * 80 + "\n")

    print(f"Finished writing tab completion scripts to {args.output_folder}")
    print("Now you can run the following to add the scripts to your zsh completion:")
    print("~" * 80)
    print(f"fpath+={args.output_folder}")
    print("autoload -Uz compinit && compinit")
    print("~" * 80)


if __name__ == "__main__":
    main()
