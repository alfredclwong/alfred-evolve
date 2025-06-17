import ray

from alfred_evolve.models.data_models import Program
from alfred_evolve.pipeline.pipeline import PipelineConfig, run_pipeline


@ray.remote
def run_pipeline_task(
    parent: Program, inspirations: list[Program], cfg: PipelineConfig, api_key: str
) -> Program | Exception:
    return run_pipeline(parent, inspirations, cfg, api_key)
