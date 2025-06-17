from dataclasses import dataclass
from pathlib import Path

import alfred_evolve.utils.docker as docker_utils
from alfred_evolve.utils.str import apply_diff_search_replace


def apply_diff(parent_content: str, diff_content: str) -> str:
    return apply_diff_search_replace(parent_content, diff_content)


@dataclass(frozen=True)
class ProgramEvaluatorConfig:
    base_name: str
    image: str
    eval_file_path: Path
    cpu_limit: str
    memory_limit: str
    timeout: int


def evaluate_program(
    program_content: str, cfg: ProgramEvaluatorConfig
) -> tuple[dict[str, float], dict[str, str]]:
    name = docker_utils.start(
        base_name=cfg.base_name,
        image=cfg.image,
        memory_limit=cfg.memory_limit,
        cpu_limit=cfg.cpu_limit,
    )

    try:
        return docker_utils.run(
            container_name=name,
            program_content=program_content,
            eval_file_path=cfg.eval_file_path,
            timeout=cfg.timeout,
        )
    except Exception:
        raise
    finally:
        docker_utils.stop(name=name)
