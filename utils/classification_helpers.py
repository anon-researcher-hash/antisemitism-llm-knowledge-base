import ast
import time
import json
import re
from os.path import join
import pandas as pd
import yaml
from google.genai import types
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

from config import DATA_DIR, ALL_GROUPS


def load_prompt_template(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)
    if not isinstance(prompt_config, dict):
        raise TypeError(
            f"Expected a YAML mapping in {prompt_path}, got {type(prompt_config).__name__}"
        )
    return prompt_config



def generate_prompt(indexed_batch, prompt_config):
    return {
        "system": prompt_config["system_prompt"],
        "user": prompt_config["task"] + "\n\nTexts:\n" + json.dumps(indexed_batch, ensure_ascii=False, indent=2)
    }



def clean_response(r):
    if type(r) == str:
        if r.startswith("Yes"):
            return "Yes"
        elif r.startswith("No"):
            return "No"
        elif r.startswith("More context needed"):
            return "More context needed"
        elif r.startswith("Unsure"):
            return "Unsure"
    else:
        return r


def fix_json_string(raw_str):
    cleaned = raw_str.strip("`").replace("json\n", "").strip()
    cleaned = re.sub(r'}\s*{', '}, {', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    return cleaned


def parse_json_response(raw_response):
    cleaned_json_str = fix_json_string(raw_response)
    batch_records = json.loads(cleaned_json_str)
    return batch_records


def extract_classifications_from_file(file_name, offset=0):
    index_to_classification = {}
    index_to_explanation = {}

    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)

        for batch in data:
            try:
                batch_records = parse_json_response(batch["raw_response"])
                if not isinstance(batch_records, list):
                    batch_records = [batch_records]
                for i in batch_records:
                    index_to_classification[int(i["index"]) + offset] = i["classification"]
                    index_to_explanation[int(i["index"]) + offset] = i["explanation"]
            except json.JSONDecodeError as e:
                print(f"Error: {e.msg} at line {e.lineno}, column {e.colno}, char {e.pos} in file {file_name}")

                raw = batch.get("raw_response", "")
                start = max(e.pos - 40, 0)
                end = min(e.pos + 40, len(raw))
                context = raw[start:end]

                print("\nContext around error:")
                print(context)
                print(" " * (e.pos - start) + "^")
                continue
            except AttributeError as e:
                print(e)
                continue
    c_df = pd.DataFrame.from_dict(index_to_classification.items())
    # print(c_df.columns)
    c_df.columns = ["index", "classification"]

    e_df = pd.DataFrame.from_dict(index_to_explanation.items())
    e_df.columns = ["index", "explanation"]

    c_e_df = pd.merge(c_df, e_df, on="index", how="inner")
    c_e_df.set_index("index", inplace=True)
    return c_e_df


def extract_classifications_from_flat_file(file_name, offset=0):
    index_to_classification = {}

    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)

        for row in data:
            try:
                idx = int(row["batch"]) + offset
                classification = str(row["raw_response"]).strip()
                index_to_classification[idx] = classification

            except KeyError as e:
                print(f"Missing key {e} in row: {row}")
                continue
            except (ValueError, TypeError, AttributeError) as e:
                print(f"Error parsing row {row}: {e}")
                continue

    c_df = pd.DataFrame.from_dict(index_to_classification.items())
    c_df.columns = ["index", "classification"]
    c_df.set_index("index", inplace=True)
    return c_df


def import_lexicon_classification_files(source, label, total, verbose=True, kb="lexicon", provider="gemini", flat_file=False, sort_index=True):
    all_records = []
    file_names = [join(DATA_DIR, "model_outputs", build_file_name(source=source, kb=kb, label=label, start=start, provider=provider)) for start in range(0, total, 50)]

    for f in file_names:
        if verbose:
            print(f)
        if not flat_file:
            r = extract_classifications_from_file(f)
        else:
            r = extract_classifications_from_flat_file(f)
        if verbose:
            print(len(r))
        all_records.append(r)
    all_records = pd.concat(all_records)
    if sort_index:
        return all_records.sort_index()
    else:
        return all_records

def extract_lex_ref_gemini(text: str) -> list[int]:
    return extract_lex_ref_gemini_1(text) + extract_lex_ref_gemini_2(text)

def extract_lex_ref_gemini_1(text):
    pattern = r'Chapter(?:s)?\s*((?:\d+(?:\.\d+)?)(?:\s*[-–—]\s*\d+(?:\.\d+)?|\s*,\s*\d+(?:\.\d+)?)*)'

    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    chapters = []

    for match in matches:
        match = re.sub(r'[–—]', '-', match)
        parts = re.split(r'[,\s]+', match.strip())
        for part in parts:
            if not part:
                continue
            part = part.strip().strip('.,;:)(')

            if re.fullmatch(r'\d+(?:\.\d+)?', part):
                chapters.append(int(part.split('.')[0]))
            elif '-' in part:
                start_str, end_str = [p.strip().strip('.,;:)(') for p in part.split('-', 1)]
                if start_str and end_str and re.fullmatch(r'\d+(?:\.\d+)?', start_str) and re.fullmatch(
                        r'\d+(?:\.\d+)?', end_str):
                    start_sec = int(start_str.split('.')[0])
                    end_sec = int(end_str.split('.')[0])
                    chapters.append(range(start_sec, end_sec + 1))

    return list(dict.fromkeys(chapters))


def extract_lex_ref_gemini_2(text: str) -> list[int]:

    chapters = []

    clause_pattern = re.compile(
        r"""(?ix)
        (?:referencing\s+relevant\s+chapters?|relevant\s+chapters?|relevant\s+chapter)
        \s*(?:include(?:s)?|:)?   
        \s*
        (.+?)                      
        (?=(?:\.\s|$))             
        """
    )

    for clause in clause_pattern.findall(text):
        clause = re.sub(r'[–—]', '-', clause)

        token_pattern = re.compile(
            r"""(?ix)
            \b(
                \d+(?:\.\d+)?            
                (?:\s*-\s*\d+(?:\.\d+)?)? 
            )
            \b
            """
        )

        tokens = token_pattern.findall(clause)

        for tok in tokens:
            tok = tok.strip().strip('.,;:)(')

            if '-' in tok:
                left, right = [p.strip() for p in tok.split('-', 1)]
                if re.fullmatch(r'\d+(?:\.\d+)?', left) and re.fullmatch(r'\d+(?:\.\d+)?', right):
                    start_sec = int(left.split('.')[0])
                    end_sec = int(right.split('.')[0])
                    if start_sec <= end_sec:
                        chapters.append(range(start_sec, end_sec + 1))
                    else:
                        chapters.append(range(end_sec, start_sec + 1))
            else:
                if re.fullmatch(r'\d+(?:\.\d+)?', tok):
                    chapters.append(int(tok.split('.')[0]))

    return list(dict.fromkeys(chapters))

def extract_tax_ref_gemini(explanation: str):
    pattern = r"""
        (?:
            \b(?P<num1>[2-9]|[1-3][0-9]|4[0-2])(?=[.:])             
            |
            \((?P<num2>[2-9]|[1-3][0-9]|4[0-2])\)                   
            |
            \b(?:concept|category|section|chapter)\s+
               (?P<num3>[2-9]|[1-3][0-9]|4[0-2])\b                  
            |
            \b(?P<num4>[2-9]|[1-3][0-9]|4[0-2])\s*\([^)]{2,}\)      
            |
            (?<=-\s)(?P<num5>[2-9]|[1-3][0-9]|4[0-2])\)   # - 29)
        )
    """

    seen = set()
    refs = []

    for match in re.finditer(pattern, explanation, re.VERBOSE | re.IGNORECASE):
        num = match.group("num1") or match.group("num2") or match.group("num3") or match.group("num4") or match.group("num5")
        num = int(num)
        if num not in seen:
            seen.add(num)
            refs.append(num)

    return refs

def find_answers_claude(text):
    yes1 = re.findall(r"\*\*\d+:\*\* Yes", text)
    yes2 = re.findall(r"\*\*\d+\*\*: Yes", text)
    no1 = re.findall(r"\*\*\d+:\*\* No", text)
    no2 = re.findall(r"\*\*\d+\*\*: No", text)
    no3 = re.findall(r"\*\*\d+:\*\* `No`", text)
    no4 = re.findall(r"\*\*Post \d+:\*\* No", text)

    yes_found = bool(yes1 or yes2)
    no_found = bool(no1 or no2 or no3 or no4)
    
    if yes_found and not no_found:
        return "Yes"
    elif no_found and not yes_found:
        return "No"
    elif yes_found and no_found:
        return "Mixed"
    return text

def extract_ihra_ref_claude(explanation: str) -> list[int]:
    seen = set()
    refs = []

    def _add(n: int):
        if 0 <= n <= 11 and n not in seen:  # 0 is valid for Claude output
            seen.add(n)
            refs.append(n)

    SEP = r'(?:\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+)'
    NUM_OR_RANGE = r'\d+(?:\s*[-–—]\s*\d+)?'
    range_or_num = re.compile(r'(\d+)\s*[-–—]\s*(\d+)|(\d+)')

    def _parse_list(text: str):
        for tok in range_or_num.finditer(text):
            if tok.group(1):
                start, end = int(tok.group(1)), int(tok.group(2))
                if start > end:
                    start, end = end, start
                for n in range(start, end + 1):
                    _add(n)
            else:
                _add(int(tok.group(3)))

    list_pattern = re.compile(
        rf'\bsections?\s+({NUM_OR_RANGE}(?:{SEP}{NUM_OR_RANGE})*)',
        re.IGNORECASE
    )
    for m in list_pattern.finditer(explanation):
        _parse_list(m.group(1))

    bracket_pattern = re.compile(r'\[(\d+)\]')
    for m in bracket_pattern.finditer(explanation):
        _add(int(m.group(1)))

    cat_pattern = re.compile(
        rf'\bcategor(?:y|ies)\s+(?:such\s+as\s+|like\s+)?({NUM_OR_RANGE}(?:{SEP}{NUM_OR_RANGE})*)',
        re.IGNORECASE
    )
    for m in cat_pattern.finditer(explanation):
        _parse_list(m.group(1))

    hash_pattern = re.compile(
        rf'#({NUM_OR_RANGE}(?:{SEP}#{NUM_OR_RANGE})*)',
        re.IGNORECASE
    )
    for m in hash_pattern.finditer(explanation):
        _parse_list(m.group(1))

    return refs



def extract_tax_ref_claude(explanation: str) -> list[int]:
    NUM = r"(?:[2-9]|[1-3][0-9]|4[0-2])"

    pattern = re.compile(
        rf"""
        (?:
            \b(?:(?:taxonomy\s+)?categor(?:y|ies))\s+
            (?P<num1>{NUM})\b
        )
        |
        (?:
            \b(?P<num2>{NUM})\s+\([A-Z][^)]+\)
        )
        |
        (?:
            \b(?P<num3>{NUM})(?=[.:])
        )
        |
        (?:
            \((?P<num5>{NUM})\s*[-–]\s*[A-Z][^)]+\)
        )
        |
        (?:
            \((?P<num4>{NUM})\)
        )
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    seen: dict[int, None] = {}
    for match in pattern.finditer(explanation):
        num_str = (
            match.group("num1")
            or match.group("num2")
            or match.group("num3")
            or match.group("num5")
            or match.group("num4")
        )
        if num_str:
            num = int(num_str)
            if num not in seen:
                seen[num] = None

    return list(seen.keys())

def extract_tax_ref_llama(explanation: str) -> list[int]:
    NUM = r"(?:[2-9]|[1-3][0-9]|4[0-2])"

    pattern = re.compile(
        rf"""
        (?:
            \b(?:(?:taxonomy\s+)?categor(?:y|ies))\s+
            (?P<num1>{NUM})\b
        )
        |
        (?:
            \b(?P<num2>{NUM})\s+\([A-Z][^)]+\)
        )
        |
        (?:
            \b(?P<num3>{NUM})(?=[.:])
        )
        |
        (?:
            \((?P<num4>{NUM})\)
        )
        |
        (?:
            \#(?P<num6>{NUM})\b
        )
        |
        (?:
            (?<=[A-Za-z])\s*[-–]\s*(?P<num7>{NUM})\b
        )
        |
        (?:
            \b(?P<num8>{NUM})\s+'[A-Z][^']*'
        )
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    seen: dict[int, None] = {}
    for match in pattern.finditer(explanation):
        num_str = (
            match.group("num1")
            or match.group("num2")
            or match.group("num3")
            or match.group("num4")
            or match.group("num6")
            or match.group("num7")
            or match.group("num8")
        )
        if num_str:
            num = int(num_str)
            if num not in seen:
                seen[num] = None

    return list(seen.keys())

def extract_ihra_ref_llama(explanation: str) -> list[int]:
    SEC = r"(?:[0-9]|1[01])"  # 0–11

    pattern = re.compile(
        rf"""
        (?:
            \[?\bsection\s+(?P<num1>{SEC})\b\]?
        )
        |
        (?:
            \b(?P<num2>{SEC})\s+\([A-Z][^)]+\)
        )
        |
        (?:
            \band\s+(?P<num3>{SEC})\b
        )
        |
        (?:
            ,\s*(?P<num4>{SEC})\b
        )
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    seen: dict[int, None] = {}
    for match in pattern.finditer(explanation):
        num_str = (
            match.group("num1")
            or match.group("num2")
            or match.group("num3")
            or match.group("num4")
        )
        if num_str:
            num = int(num_str)
            if num not in seen:
                seen[num] = None

    return list(seen.keys())

def extract_ihra_ref_gpt(explanation: str) -> list[int]:
    seen = set()
    refs = []

    def _add(n: int):
        if n not in seen:
            seen.add(n)
            refs.append(n)

    SEP = r'(?:\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+)'
    NUM_OR_RANGE = r'\d+(?:\s*[-–—]\s*\d+)?'
    range_or_num = re.compile(r'(\d+)\s*[-–—]\s*(\d+)|(\d+)')

    def _parse_list(text: str):
        for tok in range_or_num.finditer(text):
            if tok.group(1):
                start, end = int(tok.group(1)), int(tok.group(2))
                if start > end:
                    start, end = end, start
                for n in range(start, end + 1):
                    _add(n)
            else:
                _add(int(tok.group(3)))

    list_pattern = re.compile(
        rf'\bsections?\s+({NUM_OR_RANGE}(?:{SEP}{NUM_OR_RANGE})*)',
        re.IGNORECASE
    )
    for m in list_pattern.finditer(explanation):
        _parse_list(m.group(1))

    bracket_pattern = re.compile(r'\[(\d+)\]')
    for m in bracket_pattern.finditer(explanation):
        _add(int(m.group(1)))

    cat_pattern = re.compile(
        rf'\bcategor(?:y|ies)\s+(?:such\s+as\s+|like\s+)?({NUM_OR_RANGE}(?:{SEP}{NUM_OR_RANGE})*)',
        re.IGNORECASE
    )
    for m in cat_pattern.finditer(explanation):
        _parse_list(m.group(1))

    hash_pattern = re.compile(
        rf'#({NUM_OR_RANGE}(?:{SEP}#{NUM_OR_RANGE})*)',
        re.IGNORECASE
    )
    for m in hash_pattern.finditer(explanation):
        _parse_list(m.group(1))

    return refs

def extract_tax_ref_gpt(explanation: str) -> list[int]:
    seen = set()
    refs = []

    def _add(n: int):
        if 2 <= n <= 42 and n not in seen:
            seen.add(n)
            refs.append(n)

    NUM = r'(?:[2-9]|[1-3][0-9]|4[0-2])'
    SEP = r'(?:\s*[,;]\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+)'
    NUM_OR_RANGE = rf'{NUM}(?:\s*[-–—]\s*{NUM})?'
    range_or_num = re.compile(rf'({NUM})\s*[-–—]\s*({NUM})|({NUM})')

    def _parse_list(text: str):
        for tok in range_or_num.finditer(text):
            if tok.group(1):
                start, end = int(tok.group(1)), int(tok.group(2))
                if start > end:
                    start, end = end, start
                for n in range(start, end + 1):
                    _add(n)
            else:
                _add(int(tok.group(3)))

    cat_pattern = re.compile(
        rf'\bcategor(?:y|ies)\s+(?:such\s+as\s+|like\s+)?({NUM_OR_RANGE}(?:{SEP}{NUM_OR_RANGE})*)',
        re.IGNORECASE
    )
    for m in cat_pattern.finditer(explanation):
        _parse_list(m.group(1))

    dot_colon = re.compile(rf'\b({NUM})(?=[.:])')
    for m in dot_colon.finditer(explanation):
        _add(int(m.group(1)))

    paren_prefix = re.compile(rf'\(({NUM})\)')
    for m in paren_prefix.finditer(explanation):
        _add(int(m.group(1)))

    keyword = re.compile(rf'\b(?:concept|category|section|chapter)\s+({NUM})\b', re.IGNORECASE)
    for m in keyword.finditer(explanation):
        _add(int(m.group(1)))

    num_paren = re.compile(rf'\b({NUM})\s*\([^)][^)]+\)')
    for m in num_paren.finditer(explanation):
        _add(int(m.group(1)))

    standalone_range = re.compile(rf'\b({NUM})\s*[-–—]\s*({NUM})\b')
    for m in standalone_range.finditer(explanation):
        start, end = int(m.group(1)), int(m.group(2))
        if start > end:
            start, end = end, start
        for n in range(start, end + 1):
            _add(n)

    hash_pattern = re.compile(
        rf'#({NUM_OR_RANGE}(?:{SEP}#{NUM_OR_RANGE})*)',
        re.IGNORECASE
    )
    for m in hash_pattern.finditer(explanation):
        _parse_list(m.group(1))

    num_first_in_parens = re.compile(rf'\(({NUM})\s+[A-Z][^)]+\)', re.IGNORECASE)
    for m in num_first_in_parens.finditer(explanation):
        _parse_list(m.group(1))

    num_after_quote = re.compile(rf"['\"][^'\"]+['\"],?\s*({NUM})\s*\)")
    for m in num_after_quote.finditer(explanation):
        _add(int(m.group(1)))

    bare_nums_in_parens = re.compile(rf'\(({NUM}(?:\s*,\s*{NUM})+)\)')
    for m in bare_nums_in_parens.finditer(explanation):
        _parse_list(m.group(1))

    slash_separated = re.compile(rf'\b({NUM})(?:/{NUM})+\b')
    for m in slash_separated.finditer(explanation):
        full = explanation[m.start():m.end()]
        for n in re.findall(rf'\b{NUM}\b', full):
            _add(int(n))

    eg_pattern = re.compile(
        rf'e\.g\.,\s+({NUM}\s+\w[^.;]*)',
        re.IGNORECASE
    )
    for m in eg_pattern.finditer(explanation):
        _parse_list(m.group(1))

    return refs


def group_ihra_content(item):
    if item in [0, 1]:
        return ("aggressive")
    elif item in [2, 3, 6]:
        return ("classic_power")
    elif item in [4, 5, 11]:
        return ("second_postholocaust")
    elif item in [7, 8, 9, 10]:
        return ("israel")
    else:
        return None


def group_lexicon_content(chapter_str):
    lexicon_chapters_groups = []
    try:
        chapter_list = ast.literal_eval(chapter_str) if isinstance(chapter_str, str) else chapter_str
    except:
        chapter_list = []

    if len(chapter_list) == 0:
        return []

    try:
        chapter_list = [int(num) for num in chapter_list]
    except:
        return []

    for ch in chapter_list:
        if 2 <= ch <= 15:
            lexicon_chapters_groups.append("classic_power")
        elif 16 <= ch <= 27:
            lexicon_chapters_groups.append("second_postholocaust")
        elif 28 <= ch <= 36:
            lexicon_chapters_groups.append("israel")
        elif 37 <= ch <= 42:
            lexicon_chapters_groups.append("aggressive")
    return list(set(lexicon_chapters_groups))


def extract_ihra_ref_gemini(text: str) -> list[int]:
    return extract_ihra_ref_gemini_1(text) + extract_ihra_ref_gemini_2(text)

def extract_ihra_ref_gemini_1(text: str) -> list[int]:
    sections = []

    clause_pattern = re.compile(
        r"""(?ix)
        \bSections?\b
        \s*(?:include(?:s)?|are|:)?  
        \s*
        (                             
            [^\.\)]*                  
        )
        """
    )

    numtok_pattern = re.compile(
        r"""(?ix)
        \b
        \d+(?:\.\d+)?                 # start (e.g., 11 or 11.2)
        (?:\s*[-–—]\s*\d+(?:\.\d+)?)? # optional range (e.g., -13 or -11.4)
        \b
        """
    )

    for clause in clause_pattern.findall(text):
        clause = re.sub(r'[–—]', '-', clause)

        for tok in numtok_pattern.findall(clause):
            tok = tok.strip().strip('.,;:)(')
            if '-' in tok:
                left, right = [p.strip() for p in tok.split('-', 1)]
                if re.fullmatch(r'\d+(?:\.\d+)?', left) and re.fullmatch(r'\d+(?:\.\d+)?', right):
                    a, b = int(left.split('.')[0]), int(right.split('.')[0])
                    if a <= b:
                        sections.append(range(a, b + 1))
                    else:
                        sections.append(range(b, a + 1))
            else:
                if re.fullmatch(r'\d+(?:\.\d+)?', tok):
                    sections.append(int(tok.split('.')[0]))

    return list(dict.fromkeys(sections))


def extract_ihra_ref_gemini_2(text: str) -> list[int]:
    sections = []

    clause_pattern = re.compile(
        r"""(?ix)
        (?:referencing\s+relevant\s+sections?|relevant\s+sections?|relevant\s+section)
        \s*(?:include(?:s)?|:)?    # optional 'include'/'includes' or ':'
        \s*
        (.+?)                      # capture the following clause
        (?=(?:\.\s|$))             # stop at the next sentence end or string end
        """
    )

    for clause in clause_pattern.findall(text):
        clause = re.sub(r'[–—]', '-', clause)

        token_pattern = re.compile(
            r"""(?ix)
            \b(
                \d+(?:\.\d+)?            
                (?:\s*-\s*\d+(?:\.\d+)?)? 
            )
            \b
            """
        )

        tokens = token_pattern.findall(clause)

        for tok in tokens:
            tok = tok.strip().strip('.,;:)(')

            if '-' in tok:  # range like 9.1-11.2
                left, right = [p.strip() for p in tok.split('-', 1)]
                if re.fullmatch(r'\d+(?:\.\d+)?', left) and re.fullmatch(r'\d+(?:\.\d+)?', right):
                    start_sec = int(left.split('.')[0])
                    end_sec = int(right.split('.')[0])
                    if start_sec <= end_sec:
                        sections.append(range(start_sec, end_sec + 1))
                    else:
                        sections.append(range(end_sec, start_sec + 1))
            else:
                if re.fullmatch(r'\d+(?:\.\d+)?', tok):
                    sections.append(int(tok.split('.')[0]))

    return list(dict.fromkeys(sections))


def build_file_name(source, provider, kb, label, start=None, ending=".json"):
    base_path = f"{provider}_{source}_{kb}_{label}"
    if start is not None:
        base_path += f"_batch_{str(start)}"
    return base_path + ending


def map_lexicon_chapters_to_ihra_sections(chapters, mapping_dict):
    sections = []
    if type(chapters) == list:
        for c in chapters:
            if c in mapping_dict:
                sections.extend(mapping_dict[c])
        sections = [int(item) for item in sections if item is not None]
        return list(dict.fromkeys(sections))
    else:
        return chapters


def expand_ranges(section_list):
    expanded = []
    for item in section_list:
        if isinstance(item, range) or isinstance(item, list) or isinstance(item, tuple):
            expanded.extend(list(item))
        else:
            expanded.append(item)
    return list(dict.fromkeys(expanded))


def map_decoding_codes(codes, mapping_dict):
    chapters = []
    if type(codes) == list:
        for c in codes:
            if c in mapping_dict:
                chapters.append(mapping_dict[c])
        chapters = [int(item) for item in chapters if item is not None and item != "X"]
        return sorted(list(set(chapters)))
    else:
        return codes


def split_ambiguous_sections(ihra_sections):
    ihra_sections=set(ihra_sections)
    if 27 in ihra_sections:
        ihra_sections.remove(27)
        ihra_sections.add(2)
        ihra_sections.add(7)
    if 29 in ihra_sections:
        ihra_sections.remove(29)
        ihra_sections.add(2)
        ihra_sections.add(9)
    return sorted(list(ihra_sections))

def flatten(nested, interpret_as_int=True):
    flattened = []
    for section in nested:
        flattened.extend(section)
    if interpret_as_int:
        return [int(i) for i in flattened]
    return flattened

def count_items(series, interpret_as_int=True):
    series_flat = flatten(series.values, interpret_as_int=interpret_as_int)
    series_count = Counter(series_flat)
    series_count_norm = {key: np.round(value / sum(series_count.values()), 2) for key, value in series_count.items()}
    return series_count, series_count_norm

def compute_multilabel_prec_recall(df, annotator_group_col, model_group_col, groups=ALL_GROUPS):
    results = []
    for group in groups:
        annotators_has = df[annotator_group_col].apply(lambda x: group in x)
        model_has = df[model_group_col].apply(lambda x: group in x)
        both_have = annotators_has & model_has
        
        recall = both_have.sum() / annotators_has.sum() if annotators_has.sum() > 0 else 0        
        precision = both_have.sum() / model_has.sum() if model_has.sum() > 0 else 0
        
        results.append({
            'content_group': group,
            'annotators_identified': annotators_has.sum(),
            'model_identified': model_has.sum(),
            'both_identified': both_have.sum(),
            'recall': recall,
            'precision': precision
        })

    overview_df = pd.DataFrame(results)
    overview_df = overview_df.sort_values('content_group')
    return overview_df

def plot_distribution_diff_pair(annotator_counts, explanation_counts_1, explanation_counts_2, explanation_reduction_1, explanation_reduction_2, title, x_label, normalized=True):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    ylabel_text = 'Normalized Count' if normalized else 'Count'
    axs[0].bar(annotator_counts.keys(), annotator_counts.values(), alpha=0.5, label='Annotators')
    axs[0].bar(explanation_counts_1.keys(), explanation_counts_1.values(), alpha=0.5, label=explanation_reduction_1)
    axs[0].set_ylabel(ylabel_text)
    axs[0].set_xlabel(x_label)
    axs[0].legend()

    axs[1].bar(annotator_counts.keys(), annotator_counts.values(), alpha=0.5, label='Annotators')
    axs[1].bar(explanation_counts_2.keys(), explanation_counts_2.values(), alpha=0.5, label=explanation_reduction_2)
    axs[1].set_xlabel(x_label)
    axs[1].legend()
    fig.suptitle(title)

    plt.show()

def plot_distribution_diff(annotator_counts, explanation_counts, title, normalized=True):
    plt.bar(annotator_counts.keys(), annotator_counts.values(), alpha=0.5, label='Annotators')
    plt.bar(explanation_counts.keys(), explanation_counts.values(), alpha=0.5, label='Model Explanations')
    plt.xlabel('References')
    if normalized:
        plt.ylabel('Normalized Count')
    else:
        plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.show()  