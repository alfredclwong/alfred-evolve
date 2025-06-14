from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from alfred_evolve.eval.docker import run, start, stop


@dataclass(frozen=True)
class EvaluatorConfig:
    container_name: str
    image: str
    eval_file: Path
    cpu_limit: str
    memory_limit: str
    timeout: int


class Evaluator:
    def __init__(self, cfg: EvaluatorConfig):
        self.cfg = cfg

    def evaluate(self, program_content: str) -> tuple[dict[str, float], Optional[str]]:
        name = start(
            base_name=self.cfg.container_name,
            image=self.cfg.image,
            memory_limit=self.cfg.memory_limit,
            cpu_limit=self.cfg.cpu_limit,
        )
        score_dict = {"SCORE": 0.0}
        artifacts = ""
        # score_dict["COMPLEXITY"] = len(program_content) / 10000
        try:
            score_dict, artifacts = run(
                name=name,
                program_content=program_content,
                eval_file=self.cfg.eval_file,
                timeout=self.cfg.timeout,
            )
        except Exception as e:
            print(f"Error during evaluation: {e}")
        finally:
            stop(name=name)
            return score_dict, artifacts
