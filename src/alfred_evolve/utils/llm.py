import json

import requests

from alfred_evolve.utils.logging import get_logger

logger = get_logger(__name__)


def get_llm_response(
    prompt: str,
    api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    cost_in: float,
    cost_out: float,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        llm_output = response_data["choices"][0]["message"]["content"]
        tokens_in = response_data["usage"]["prompt_tokens"]
        tokens_out = response_data["usage"]["completion_tokens"]
        cost = tokens_in * cost_in + tokens_out * cost_out
        logger.info(f"LLM response cost: ${cost:.4f} (in: {tokens_in}, out: {tokens_out})")
        return llm_output
    else:
        error_message = f"Error: {response.status_code} - {response.text}"
        raise Exception(error_message)
