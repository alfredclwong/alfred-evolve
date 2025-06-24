import google.cloud.logging
from google.api_core.exceptions import AlreadyExists
from google.cloud import run_v2
from google.protobuf.duration_pb2 import Duration

from alfred_evolve.utils.logging import get_logger

logger = get_logger(__name__)


def get_parent_name(project_id: str, region: str) -> str:
    return f"projects/{project_id}/locations/{region}"


def get_job_id(base_name: str, cpu_limit: str, memory_limit: str) -> str:
    return f"{base_name}-{cpu_limit}cpu-{memory_limit.lower()}"


def get_job_name(
    project_id: str, region: str, base_name: str, cpu_limit: str, memory_limit: str
) -> str:
    parent = get_parent_name(project_id, region)
    job_id = get_job_id(base_name, cpu_limit, memory_limit)
    return f"{parent}/jobs/{job_id}"


def create_job(
    image: str,
    cpu_limit: str,
    memory_limit: str,
    project_id: str,
    region: str,
    base_name: str,
) -> str:
    jobs_client = run_v2.JobsClient()
    job = run_v2.Job(
        template=run_v2.ExecutionTemplate(
            template=run_v2.TaskTemplate(
                containers=[
                    run_v2.Container(
                        image=image,
                        resources=run_v2.ResourceRequirements(
                            limits={
                                "cpu": cpu_limit,
                                "memory": memory_limit,
                            }
                        ),
                    )
                ],
                max_retries=0,
            )
        )
    )
    parent = get_parent_name(project_id, region)
    job_id = get_job_id(base_name, cpu_limit, memory_limit)
    create_job_request = run_v2.CreateJobRequest(
        parent=parent,
        job_id=job_id,
        job=job,
    )
    try:
        create_job_operation = jobs_client.create_job(create_job_request)
        create_job_result = create_job_operation.result()
        job_name = getattr(create_job_result, "name")
        logger.info(f"Created job: {job_name}")
    except AlreadyExists:
        job_name = get_job_name(project_id, region, base_name, cpu_limit, memory_limit)
        logger.info(f"Using existing job: {job_name}")
    return job_name


def run_job(job_name: str, env_vars: dict[str, str], timeout: int) -> str:
    jobs_client = run_v2.JobsClient()
    run_job_request = run_v2.RunJobRequest(
        name=job_name,
        overrides=run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(
                    env=[
                        run_v2.EnvVar(name=key, value=value)
                        for key, value in env_vars.items()
                    ],
                ),
            ],
            timeout=Duration(seconds=timeout * 2),  # Give some buffer for execution startup
        ),
    )
    run_job_operation = jobs_client.run_job(run_job_request)
    run_job_result = run_job_operation.result()
    exec_name = getattr(run_job_result, "name").split("/")[-1]
    logger.info(f"Finished job execution: {exec_name}")
    return exec_name


def get_result(exec_name: str) -> str:
    logging_client = google.cloud.logging.Client()
    query = f"""
        resource.type="cloud_run_job" AND 
        labels."run.googleapis.com/execution_name"="{exec_name}" AND
        severity="DEFAULT"
    """
    result = "\n".join(
        [
            entry.payload
            for entry in logging_client.list_entries(filter_=query)
            if isinstance(entry, google.cloud.logging.TextEntry)
        ]
    )
    logger.info(f"Retrieved logs for execution: {exec_name}")
    return result


def delete_job(job_name: str):
    jobs_client = run_v2.JobsClient()
    delete_job_request = run_v2.DeleteJobRequest(name=job_name)
    jobs_client.delete_job(delete_job_request)
    logger.info(f"Deleted job: {job_name}")
