import argparse
import json
import os
import sys
import time
from os.path import join
from pathlib import Path
from typing import Any, Callable

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

import anthropic

import openai

import google.generativeai as genai
from dotenv import load_dotenv

from utils.classification_helpers import load_prompt_template, generate_prompt, build_file_name
from config import TEXT_COL, DataSource, MODELS, PROJECT_DIR, DATA_DIR, PROMPT_FILES, MODEL_KEYS, LLAMA_URL

load_dotenv()

CHUNK_SIZE = 50
BATCH_SIZE = 1
MAX_RETRIES = 3


def build_model(model_type) -> Any:
    if model_type == "gemini":
        print(f"Loading Gemini model: {MODELS['gemini']}")
        return genai.GenerativeModel(MODELS["gemini"])
    elif model_type == "claude":
        print(f"Loading Claude model: {MODELS['claude']}")
        return anthropic.Anthropic(api_key=MODEL_KEYS["claude"])
    elif model_type == "gpt":
        print(f"Loading GPT model: {MODELS['gpt']}")
        return openai.OpenAI(api_key=MODEL_KEYS["gpt"])
    elif model_type == "llama":
        print(f"Loading Llama model: {MODELS['llama']}")
        return openai.OpenAI(
        api_key=MODEL_KEYS["llama"],
        base_url=LLAMA_URL
    )


def prompt_gemini(
        model: Any,
        prompt: dict,
        batch_number: int,
        max_retries: int = MAX_RETRIES,
) -> dict:
    retries = 0

    while retries < max_retries:
        try:
            response = model.generate_content(
                contents=[prompt["system"] + "\n\n" + prompt["user"]],
                generation_config={"temperature": 0.0}
            )

            if response.candidates and response.candidates[0].content.parts:
                return {
                    "batch": batch_number,
                    "raw_response": response.text
                }

            retries += 1
            print(
                f"Retrying batch {batch_number} "
                f"({retries}/{max_retries})..."
            )

        except Exception as e:
            retries += 1
            print(
                f"Error in batch {batch_number}: {e} "
                f"({retries}/{max_retries})"
            )
            time.sleep(2)

    print(
        f"Failed to retrieve batch {batch_number} "
        f"after {max_retries} retries"
    )
    return {
        "batch": batch_number,
        "raw_response": None
    }


def prompt_claude(
        model: Any, 
        prompt: dict, 
        batch_number: int, 
        max_retries: int = MAX_RETRIES
        ) -> dict:
    retries = 0
    while retries < max_retries:
        try:
            response = model.messages.create(
                model=MODELS["claude"],
                max_tokens=1000,  # needs to be specified. max is 64,000. 1000 corresponds twice times the longest response of Gemini
                temperature=0.0,
                system=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}]
)
            return {
                "batch": batch_number,
                "raw_response": response.content[0].text
            }
        except Exception as e:
            retries += 1
            print(f"Error in batch {batch_number}: {e} ({retries}/{max_retries})")
            time.sleep(2)

    print(f"Failed to retrieve batch {batch_number} after {max_retries} retries")
    return {"batch": batch_number, "raw_response": None}


def prompt_gpt(model: Any, prompt: dict, batch_number: int, max_retries: int = MAX_RETRIES) -> dict:
    retries = 0
    while retries < max_retries:
        try:
            response = model.chat.completions.create(
                model=MODELS["gpt"],
                temperature=0.0,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ]
            )
            return {
                "batch": batch_number,
                "raw_response": response.choices[0].message.content
            }
        except Exception as e:
            retries += 1
            print(f"Error in batch {batch_number}: {e} ({retries}/{max_retries})")
            time.sleep(2)

    print(f"Failed to retrieve batch {batch_number} after {max_retries} retries")
    return {"batch": batch_number, "raw_response": None}

def prompt_llama(model: Any, prompt: dict, batch_number: int, max_retries: int = MAX_RETRIES) -> dict:
    retries = 0
    while retries < max_retries:
        try:
            response = model.chat.completions.create(
                model=MODELS["llama"],
                temperature=0.0,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ]
            )
            return {
                "batch": batch_number,
                "raw_response": response.choices[0].message.content
            }
        except Exception as e:
            retries += 1
            print(f"Error in batch {batch_number}: {e} ({retries}/{max_retries})")
            time.sleep(2)

    print(f"Failed to retrieve batch {batch_number} after {max_retries} retries")
    return {"batch": batch_number, "raw_response": None}

def get_model_provider(provider: str) -> tuple[Any, Callable]:
    print(provider)
    model = build_model(provider)
    if provider == "gemini":
        prompt_fn = prompt_gemini
    elif provider == "claude":
        prompt_fn = prompt_claude
    elif provider == "gpt":
        prompt_fn = prompt_gpt
    elif provider == "llama":
        prompt_fn = prompt_llama

    return model, prompt_fn


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Classify posts using an LLM without caching a PDF."
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        help="Starting index for processing."
    )
    parser.add_argument(
        "--data_source", "-d",
        type=str,
        choices=[s.value for s in DataSource],
        help="Data source to process. Either 'bloomington' or 'decoding'."
    )
    parser.add_argument(
        "--label", "-l",
        type=str,
        choices=["0", "1"],
        default="1",
        help="Class/label in data."
    )
    parser.add_argument(
        "--kb", "-k",
        type=str,
        choices=PROMPT_FILES.keys(),
        help="Knowledge base to use. Either 'ihra', 'tax', 'tax_ex', or 'no_kb'."
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        choices=["gemini", "claude", "gpt", "llama"],
        default="gemini",
        help="LLM provider to use."
    )

    return parser.parse_args()


def process_batch(
        indexed_batch: dict[int, str],
        prompt_config: dict,
        model: Any,
        prompt_fn: Callable,
        batch_number: int,
) -> dict:
    prompt = generate_prompt(indexed_batch, prompt_config)
    return prompt_fn(
        model=model,
        prompt=prompt,
        batch_number=batch_number
    )


if __name__ == "__main__":
    args = parse_arguments()
    START = args.start
    source = args.data_source
    kb_name = args.kb
    kb_file = PROMPT_FILES[kb_name]
    provider = args.provider

    print(f"Data source: {source}")
    print(f"Provider: {provider}")

    texts = pd.read_feather(
        join(DATA_DIR, f"{source}_label_{args.label}_text.feather")
    )[TEXT_COL].values.tolist()

    prompt_config = load_prompt_template(
        join(PROJECT_DIR, "llm_prompting", "prompts", kb_file)
    )
    API_KEY = MODEL_KEYS[provider]
    if provider == "gemini":
        genai.configure(api_key=API_KEY) 

    model, prompt_fn = get_model_provider(provider)

    for start in range(START, len(texts), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(texts))
        chunk_texts = texts[start:end]
        print(f"Processing chunk {start} to {end}")

        results = []

        for i in range(0, len(chunk_texts), BATCH_SIZE):
            batch = chunk_texts[i:i + BATCH_SIZE]
            indexed_batch = {
                j: text for j, text in enumerate(batch, start=start + i)
            }
            batch_number = i // BATCH_SIZE + 1

            result = process_batch(
                indexed_batch=indexed_batch,
                prompt_config=prompt_config,
                model=model,
                prompt_fn=prompt_fn,
                batch_number=batch_number,
            )
            results.append(result)

        file_name = build_file_name(
            source=source,
            label=args.label,
            provider=provider,
            kb=kb_name,
            start=start
        )
        file_path = join(DATA_DIR, "model_outputs", file_name)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Saved results to {file_path}")
