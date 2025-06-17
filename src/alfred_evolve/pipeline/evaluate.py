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
    n_eval_runs: int


def evaluate_program(
    program_content: str, cfg: ProgramEvaluatorConfig
) -> tuple[dict[str, float], dict[str, str]]:
    score_dicts = []
    artifacts_dicts = []

    name = docker_utils.start(
        base_name=cfg.base_name,
        image=cfg.image,
        memory_limit=cfg.memory_limit,
        cpu_limit=cfg.cpu_limit,
    )
    try:
        for _ in range(cfg.n_eval_runs):
            score_dict, artifacts_dict = docker_utils.run(
                container_name=name,
                program_content=program_content,
                eval_file_path=cfg.eval_file_path,
                timeout=cfg.timeout,
            )
            score_dicts.append(score_dict)
            artifacts_dicts.append(artifacts_dict)
    except Exception:
        raise
    finally:
        docker_utils.stop(name=name)

    # Aggregate scores and artifacts
    aggregated_scores = {}
    aggregated_artifacts = {}
    for score_dict in score_dicts:
        for key, value in score_dict.items():
            score = aggregated_scores.get(key, float("inf"))
            aggregated_scores[key] = min(score, value)
    aggregated_scores["MAX_SCORE"] = max(
        score_dict.get("SCORE", 0.0) for score_dict in score_dicts
    )
    for i, artifacts_dict in enumerate(artifacts_dicts):
        for key, value in artifacts_dict.items():
            aggregated_artifacts[f"{key}_{i}"] = value

    return aggregated_scores, aggregated_artifacts
