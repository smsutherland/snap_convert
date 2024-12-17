from pathlib import Path
import subprocess


git_version = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
)
