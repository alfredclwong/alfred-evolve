import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from alfred_evolve.util import extract_tagged_text


@dataclass(frozen=True)
class DiffGeneratorConfig:
    api_key_path: Path
    model_name: str
    temperature: float = 0.7
    max_tokens: int | None = None


class DiffGenerator:
    def __init__(self, cfg: DiffGeneratorConfig):
        self.cfg = cfg
        self.api_key = cfg.api_key_path.read_text().strip()

    def generate(self, prompt: str) -> tuple[Optional[str], Optional[str]]:
        try:
            llm_output = self._get_llm_response(prompt)
            # llm_output = self._get_dummy_response(prompt)
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            llm_output = ""
        reasoning = extract_tagged_text(llm_output, "REASONING")
        diff = extract_tagged_text(llm_output, "DIFF")
        if not diff:
            print("No diff found in LLM output:")
            print(llm_output)
        return diff, reasoning

    def _get_dummy_response(self, prompt: str) -> str:
        """For testing purposes, returns a dummy response."""
        return (
            "<REASONING>poopoo</REASONING>"
            "<DIFF>@@ -0,0 +0,29 @@\n"
            "+import numpy as np"
            "+"
            "+def pack_26():"
            "+    circles = np.array(["
            "+        (0.10, 0.10, 0.08),"
            "+        (0.30, 0.10, 0.08),"
            "+        (0.50, 0.10, 0.08),"
            "+        (0.70, 0.10, 0.08),"
            "+        (0.90, 0.10, 0.08),"
            "+        (0.20, 0.30, 0.08),"
            "+        (0.40, 0.30, 0.08),"
            "+        (0.60, 0.30, 0.08),"
            "+        (0.80, 0.30, 0.08),"
            "+        (0.10, 0.50, 0.08),"
            "+        (0.30, 0.50, 0.08),"
            "+        (0.50, 0.50, 0.08),"
            "+        (0.70, 0.50, 0.08),"
            "+        (0.90, 0.50, 0.08),"
            "+        (0.20, 0.70, 0.08),"
            "+        (0.40, 0.70, 0.08),"
            "+        (0.60, 0.70, 0.08),"
            "+        (0.80, 0.70, 0.08),"
            "+        (0.10, 0.90, 0.08),"
            "+        (0.30, 0.90, 0.08),"
            "+        (0.50, 0.90, 0.08),"
            "+        (0.70, 0.90, 0.08),"
            "+        (0.90, 0.90, 0.08),"
            "+    ])"
            "+    return circles"
            "</DIFF>"
        )

    def _get_llm_response(self, prompt: str) -> str:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.cfg.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.cfg.temperature,
        }
        if self.cfg.max_tokens is not None:
            data["max_tokens"] = self.cfg.max_tokens

        # is_reasoning_model = "deepseek" in model_name.lower()
        # if is_reasoning_model:
        #     data["reasoning"] = {
        #         # "max_tokens": min(512, max_tokens // 4) if max_tokens else None,
        #         # "exclude": True,
        #     }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            response_data = response.json()
            llm_output = response_data["choices"][0]["message"]["content"]
            return llm_output
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
