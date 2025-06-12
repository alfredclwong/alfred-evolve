import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional
import uuid

from alfred_evolve.util import extract_tagged_text


def run(name: str, program_content: str, eval_file: Path) -> dict[str, float]:
    eval_result = _eval(name, program_content, eval_file)
    score_str = extract_tagged_text(eval_result, "SCORE")
    if not score_str:
        return {}
    score_dict = {
        k: float(v) for k, v in (item.split(": ") for item in score_str.split(", "))
    }
    return score_dict


def start(
    base_name: str,
    image: str,
    memory_limit: Optional[str] = None,
    cpu_limit: Optional[str] = None,
    timeout: int = 10,
) -> str:
    start = time.time()
    while True:
        try:
            name = f"{base_name}_{uuid.uuid4().hex[:4]}"
            args = ["docker", "run", "-it", "--rm", "--name", name]
            if memory_limit:
                args.extend(["--memory", memory_limit])
            if cpu_limit:
                args.extend(["--cpus", cpu_limit])
            args.extend(["-d", image])
            subprocess.run(args, check=True)
            _wait_for_ready(name)
            return name
        except subprocess.CalledProcessError as e:
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Failed to start Docker container {base_name} within {timeout} seconds."
                )


def stop(name: str):
    subprocess.run(["docker", "stop", name], check=True)


def _eval(name: str, program_content: str, eval_file: Path) -> str:
    eval_path = Path("eval.py")
    _cp(name, eval_file, eval_path)
    program_path = Path("program.py")
    _write(name, program_path, program_content)
    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                name,
                "python3",
                str(eval_path),
                "-p",
                str(program_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running Docker command: {e}")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error output: {e.stderr}")
        return ""


def _cp(name: str, src: Path, dest: Path):
    """Copy a file from the host to the Docker container."""
    subprocess.run(
        [
            "docker",
            "cp",
            str(src),
            f"{name}:{dest}",
        ],
        check=True,
    )
    print(f"Copied {src} to {dest} in Docker container {name}.")


def _write(name: str, docker_path: Path, completion: str):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(completion.encode("utf-8"))
    temp_file_path = Path(temp_file.name)
    _cp(name, temp_file_path, docker_path)
    temp_file_path.unlink(missing_ok=True)


def _wait_for_ready(name: str, timeout: int = 60):
    start_time = time.time()
    while True:
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", name],
                check=True,
                capture_output=True,
            )
            if result.stdout.strip() == b"true":
                return
        except subprocess.CalledProcessError:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Container {name} did not become ready within {timeout} seconds."
                )
            time.sleep(1)
