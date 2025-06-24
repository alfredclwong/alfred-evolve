import subprocess
import uuid
from dataclasses import dataclass

from alfred_evolve.utils.gcp import create_job, get_result, run_job
from alfred_evolve.utils.logging import get_logger
from alfred_evolve.utils.str import (
    apply_diff_search_replace,
    extract_tagged_text,
    parse_json,
)

logger = get_logger(__name__)


def apply_diff(parent_content: str, diff_content: str) -> str:
    return apply_diff_search_replace(parent_content, diff_content)


@dataclass(frozen=True)
class ProgramEvaluatorConfig:
    cpu_limit: str
    memory_limit: str
    timeout: int
    n_eval_runs: int
    base_name: str
    image: str


@dataclass(frozen=True)
class DockerConfig(ProgramEvaluatorConfig):
    pass


@dataclass(frozen=True)
class GoogleCloudEvaluatorConfig(ProgramEvaluatorConfig):
    region: str
    project_id: str
    image: str


def evaluate_program(
    program_content: str, cfg: ProgramEvaluatorConfig
) -> tuple[dict[str, float], dict[str, str]]:
    if isinstance(cfg, GoogleCloudEvaluatorConfig):
        eval_fn = _evaluate_program_remote
    elif isinstance(cfg, DockerConfig):
        eval_fn = _evaluate_program_local
    else:
        raise ValueError(f"Unsupported evaluator config type: {type(cfg)}")

    score_dicts = []
    artifact_dicts = []
    for _ in range(cfg.n_eval_runs):
        score_dict, artifact_dict = eval_fn(program_content, cfg)
        score_dicts.append(score_dict)
        artifact_dicts.append(artifact_dict)

    score_dict = {k: min(d[k] for d in score_dicts) for k in score_dicts[0]}
    artifact_dict = {
        f"{k}_{i}": v for i, d in enumerate(artifact_dicts) for k, v in d.items()
    }
    return score_dict, artifact_dict


def _evaluate_program_remote(
    program_content: str, cfg: GoogleCloudEvaluatorConfig
) -> tuple[dict[str, float], dict[str, str]]:
    job_name = create_job(
        cfg.image,
        cfg.cpu_limit,
        cfg.memory_limit,
        cfg.project_id,
        cfg.region,
        cfg.base_name,
    )
    env_vars = {
        "PROGRAM_CONTENT": program_content,
        "TIME_LIMIT": str(cfg.timeout),
    }
    exec_name = run_job(job_name, env_vars, cfg.timeout)
    result = get_result(exec_name)
    score_dict, artifact_dict = _parse_eval_result(result)
    return score_dict, artifact_dict


def _evaluate_program_local(
    program_content: str, cfg: DockerConfig
) -> tuple[dict[str, float], dict[str, str]]:
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            cfg.memory_limit,
            "--cpus",
            cfg.cpu_limit,
            "--name",
            f"{cfg.base_name}-{uuid.uuid4()}",
            "--env",
            f"PROGRAM_CONTENT={program_content}",
            "--env",
            f"TIME_LIMIT={cfg.timeout}",
            cfg.image,
        ],
        capture_output=True,
        text=True,
    )
    return _parse_eval_result(result.stdout)


def _parse_eval_result(result: str) -> tuple[dict[str, float], dict[str, str]]:
    score_str = extract_tagged_text(result, "SCORE")
    artifact_str = extract_tagged_text(result, "ARTIFACT")
    score_dict = parse_json(score_str)
    score_dict = {k: float(v) for k, v in score_dict.items()}
    artifact_dict = parse_json(artifact_str)
    return score_dict, artifact_dict
