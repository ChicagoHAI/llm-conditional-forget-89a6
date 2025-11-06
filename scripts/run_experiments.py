import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from anthropic import Anthropic
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "conditional_forgetting.jsonl"
OUTPUT_DIR = ROOT / "results" / "model_outputs"


@dataclass(frozen=True)
class ModelSpec:
    name: str  # human readable alias
    provider: str  # "openai" | "anthropic" | "openrouter"
    model: str
    prompt_styles: Tuple[str, ...] = ("direct", "cot")


PROMPT_STYLES: Dict[str, Dict[str, str]] = {
    "direct": {
        "instruction": (
            "You are evaluating conditional rules in hypothetical scenarios. "
            "Ignore all real-world facts and comply exactly with the rule provided."
        ),
        "postamble": "Respond with the single capital letter (A, B, C, or D) for your chosen option.",
    },
    "cot": {
        "instruction": (
            "You are evaluating conditional rules in hypothetical scenarios. "
            "Ignore all real-world facts and comply exactly with the rule provided. "
            "Think step-by-step to verify the rule before answering."
        ),
        "postamble": (
            "Explain your reasoning briefly, then provide `Final Answer: <letter>` on a new line with a single capital letter."
        ),
    },
}


def load_dataset() -> List[Dict]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_messages(scenario: Dict, prompt_style: str) -> List[Dict[str, str]]:
    style = PROMPT_STYLES[prompt_style]
    choices_text = "\n".join(f"{label}) {text}" for label, text in scenario["choices"].items())
    user_prompt = (
        f"Rule:\n{scenario['rule']}\n\n"
        f"Question:\n{scenario['question']}\n\n"
        f"Choices:\n{choices_text}\n\n"
        f"{style['postamble']}"
    )
    return [
        {"role": "system", "content": style["instruction"]},
        {"role": "user", "content": user_prompt},
    ]


LETTER_PATTERN = re.compile(r"(?:Final Answer[:\s]*)?\b([A-D])\b", flags=re.IGNORECASE)


def parse_choice(text: str) -> Optional[str]:
    matches = LETTER_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].upper()


def call_openai(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> Tuple[str, Dict]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        top_p=1,
    )
    content = response.choices[0].message.content
    usage = {
        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
        "completion_tokens": getattr(response.usage, "completion_tokens", None),
        "total_tokens": getattr(response.usage, "total_tokens", None),
    }
    return content, usage


def call_anthropic(client: Anthropic, model: str, messages: List[Dict[str, str]]) -> Tuple[str, Dict]:
    system = ""
    user_content = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        elif msg["role"] == "user":
            user_content.append({"type": "text", "text": msg["content"]})
        else:
            user_content.append({"type": "text", "text": msg["content"]})
    response = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    text_parts = [block.text for block in response.content if block.type == "text"]
    content = "\n".join(text_parts)
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return content, usage


def call_openrouter(model: str, messages: List[Dict[str, str]]) -> Tuple[str, Dict]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Source your environment with credentials.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://research-workspace.local",
        "X-Title": "Conditional Forgetting Study",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage


def ensure_client(spec: ModelSpec):
    if spec.provider == "openai":
        return OpenAI()
    if spec.provider == "anthropic":
        key = os.getenv("CLAUDE_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Anthropic key not available (set CLAUDE_KEY or ANTHROPIC_API_KEY).")
        return Anthropic(api_key=key)
    return None


def main() -> None:
    dataset = load_dataset()
    limit = os.getenv("EVAL_LIMIT")
    if limit:
        dataset = dataset[: int(limit)]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_specs = [
        ModelSpec(name="gpt-4.1", provider="openai", model="gpt-4.1"),
        ModelSpec(name="gpt-4o-mini", provider="openai", model="gpt-4o-mini"),
        ModelSpec(name="claude-3.5-sonnet", provider="openrouter", model="anthropic/claude-3.5-sonnet"),
        ModelSpec(name="mistral-large-2407", provider="openrouter", model="mistralai/mistral-large-2407"),
    ]

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")

    for spec in model_specs:
        client = ensure_client(spec)
        for prompt_style in spec.prompt_styles:
            outfile = OUTPUT_DIR / f"{run_timestamp}__{spec.name}__{prompt_style}.jsonl"
            print(f"Running {spec.name} ({prompt_style}); writing to {outfile}")

            with outfile.open("w", encoding="utf-8") as f:
                for scenario in dataset:
                    messages = build_messages(scenario, prompt_style)
                    try:
                        if spec.provider == "openai":
                            content, usage = call_openai(client, spec.model, messages)
                        elif spec.provider == "anthropic":
                            content, usage = call_anthropic(client, spec.model, messages)
                        elif spec.provider == "openrouter":
                            content, usage = call_openrouter(spec.model, messages)
                        else:
                            raise ValueError(f"Unknown provider {spec.provider}")
                    except Exception as exc:
                        record = {
                            "scenario_id": scenario["id"],
                            "domain": scenario["domain"],
                            "rule": scenario["rule"],
                            "prompt_style": prompt_style,
                            "model_name": spec.name,
                            "model_id": spec.model,
                            "raw_response": "",
                            "parsed_choice": None,
                            "correct_choice": scenario["correct_choice"],
                            "is_correct": False,
                            "error": repr(exc),
                        }
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                        print(f"[ERROR] {spec.name} {prompt_style} on {scenario['id']}: {exc}")
                        time.sleep(2)
                        continue

                    parsed_choice = parse_choice(content or "")
                    record = {
                        "scenario_id": scenario["id"],
                        "domain": scenario["domain"],
                        "rule": scenario["rule"],
                        "prompt_style": prompt_style,
                        "model_name": spec.name,
                        "model_id": spec.model,
                        "raw_response": content,
                        "parsed_choice": parsed_choice,
                        "correct_choice": scenario["correct_choice"],
                        "is_correct": parsed_choice == scenario["correct_choice"],
                        "usage": usage,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    time.sleep(0.3)  # gentle pacing to respect rate limits


if __name__ == "__main__":
    main()
