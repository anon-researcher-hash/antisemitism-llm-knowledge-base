import argparse
import datetime
import json
import os
import sys
import time
from os.path import join
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai import caching

from utils.classification_helpers import load_prompt_template, generate_prompt, build_file_name
from utils.db_helpers import fetch_data

from config import TEXT_COL, DataSource, MODELS, PROJECT_DIR, DATA_DIR

load_dotenv()

API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=API_KEY)


def build_cached_model(context_file_path: str = "../external_ressources/lexicon.pdf"):
    context = genai.upload_file(context_file_path)
    # Wait for the file to finish processing
    while context.state.name == 'PROCESSING':
        print('Waiting for document to be processed.')
        time.sleep(2)
        context = genai.get_file(context.name)
    print(f'Document upload complete: {context.uri}')
    cache = caching.CachedContent.create(
        model=MODELS["gemini"],
        display_name='decoding lexicon',  # used to identify the cache
        contents=[context],
        ttl=datetime.timedelta(minutes=30),
    )
    print("Model with cached content loaded")
    return genai.GenerativeModel.from_cached_content(cached_content=cache)


CHUNK_SIZE = 50
BATCH_SIZE = 10
MAX_RETRIES = 3


def parse_arguments():
    global args
    parser = argparse.ArgumentParser(description="Classify posts using Gemini with Lexicon as knowledge base.")
    parser.add_argument("--start", "-s", type=int, default=0,
                        help="Starting index for processing.")
    parser.add_argument("--data_source", "-d", type=str, choices=[s.value for s in DataSource],
                        help="Data source to process")
    parser.add_argument("--label", "-l", type=str, choices=['0', '1'], default='1', help="class/label in data.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to process from each source.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    START = args.start
    source = args.data_source
    print(source)

    texts = pd.read_feather(join(DATA_DIR, f"{source}_label_{args.label}_text.feather"))[TEXT_COL].values.tolist()

    prompt_config = load_prompt_template(join(PROJECT_DIR, "prompts", "lexicon.yml"))
    model = build_cached_model()
    for start in range(START, len(texts), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(texts))
        chunk_texts = texts[start:end]
        print(f"Processing chunk {start} to {end}")

        results = []

        for i in range(0, len(chunk_texts), BATCH_SIZE):
            batch = chunk_texts[i:i + BATCH_SIZE]
            indexed_batch = {j: text for j, text in enumerate(batch, start=start + i)}
            prompt = generate_prompt(indexed_batch, prompt_config)

            retries = 0
            while retries < MAX_RETRIES:
                response = model.generate_content(contents=[prompt], generation_config={"temperature": 0.0})
                if response.candidates and response.candidates[0].content.parts:
                    results.append({
                        "batch": i // BATCH_SIZE + 1,
                        "raw_response": response.text
                    })
                    break
                else:
                    retries += 1
                    print(f"Retrying batch {i // BATCH_SIZE + 1} ({retries}/{MAX_RETRIES})...")

            if retries == MAX_RETRIES:
                print(f"Failed to retrieve batch {i // BATCH_SIZE + 1} after {MAX_RETRIES} retries")
                results.append({
                    "batch": i // BATCH_SIZE + 1,
                    "raw_response": None
                })

        # Save raw results
        file_name = build_file_name(source=source, label=args.label, kb="lexicon", start=start, ending="json")
        file_path = join("../classification/data", "gemini2-new-prompt", file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
