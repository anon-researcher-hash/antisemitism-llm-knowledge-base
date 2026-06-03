from enum import Enum
from os.path import join, dirname, abspath
from dotenv import load_dotenv
import os
load_dotenv()

PROJECT_DIR = dirname(abspath(__file__))
DATA_DIR = join(PROJECT_DIR, "data")
PROMPT_DIR = join(PROJECT_DIR, "prompts")
OUTPUT_DIR = join(DATA_DIR, "classification_outputs")

MODELS = {
    "gemini": "models/gemini-2.5-flash",
    "gpt": "gpt-5.4",
    "claude": "claude-sonnet-4-6",
    "llama": "llama-3.3-70b-instruct"
}

MODEL_KEYS = {
    "gemini": os.getenv("GOOGLE_API_KEY"),
    "gpt": os.getenv("OPENAI_API_KEY"),
    "claude": os.getenv("CLAUDE_API_KEY"),
    "llama": os.getenv("GU_API_KEY")
}

LLAMA_URL = os.getenv("LLAMA_URL")

TABLE_DEC = "decoding"
TABLE_BLO = "bloomington"
TEXT_COL = "comment_cleaned"


class DataSource(Enum):
    BLOOMINGTON = "bloomington"
    DECODING = "decoding"


PROMPT_FILES = {
    "ihra": "ihra.yml",
    "tax": "lexicon_taxonomy.yml",
    "tax_ex": "lexicon_taxonomy_examples.yml",
    "no_kb": "no_kb.yml"}


CLASS_COLS = [
    "classification_no_kb_cleaned",
    "classification_ihra_explanation_cleaned",
    #"classification_lexicon",
    "classification_tax",
    "classification_tax_ex"]

EXP_COLS = [
    "explanation_ihra_explanation_cleaned",
    #"explanation_lexicon",
    "explanation_tax",
    "explanation_tax_ex"
]

ALL_GROUPS = {'classic_power', 'aggressive', 'second_postholocaust', 'israel'}