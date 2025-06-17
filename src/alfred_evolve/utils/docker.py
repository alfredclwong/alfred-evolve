import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from alfred_evolve.utils.str import extract_tagged_text, parse_json


def run(
    program_content: str, container_name: str, eval_file_path: Path, timeout: int
) -> tuple[dict[str, float], dict[str, str]]:
    eval_result = _eval(container_name, program_content, eval_file_path, timeout)
    score_str = extract_tagged_text(eval_result, "SCORE")
    score_dict = parse_json(score_str)
    score_dict = {k: float(v) for k, v in score_dict.items()}
    artifacts_str = extract_tagged_text(eval_result, "ARTIFACT")
    artifacts_dict = parse_json(artifacts_str)
    return score_dict, artifacts_dict


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
            name = f"{base_name}_{uuid.uuid4()}"
            args = ["docker", "run", "-it", "--rm", "--name", name]
            if memory_limit:
                args.extend(["--memory", memory_limit])
            if cpu_limit:
                args.extend(["--cpus", cpu_limit])
            args.extend(["-d", image])
            subprocess.run(args, check=True)
            _wait_for_ready(name)
            return name
        except subprocess.CalledProcessError:
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Failed to start Docker container {base_name} within {timeout} seconds."
                )


def stop(name: str):
    subprocess.run(["docker", "stop", name], check=True)


def _eval(name: str, program_content: str, eval_file_path: Path, timeout: int) -> str:
    docker_eval_file_path = Path("eval.py")
    _cp(name, eval_file_path, docker_eval_file_path)
    program_path = Path("program.py")
    _write(name, program_path, program_content)
    result_str = ""
    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                name,
                "python3",
                str(docker_eval_file_path),
                "-p",
                str(program_path),
                "-t",
                str(timeout),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result_str = result.stdout
    except subprocess.CalledProcessError as e:
        result = (
            "<ARTIFACT>"
            f"docker_error: {e}\n"
            f"docker_cmd: {e.cmd}\n"
            f"docker_out: {e.output}\n"
            "</ARTIFACT>"
        )
    finally:
        return result_str


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
    # print(f"Copied {src} to {dest} in Docker container {name}.")


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
