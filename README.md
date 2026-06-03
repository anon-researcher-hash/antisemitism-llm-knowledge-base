# Anonymous Code Repository

This repository accompanies a submission with the title ``You Frame It: How Conceptual Representations Shape LLM Detection and Reasoning about Antisemitism'', currently under review.
It contains code only and is shared for the purpose of supporting anonymous peer review.

## Overview

The repository provides the codebase used for:

- Cleaning and preparation of input data
- Large Language Model (LLM) prompting
- Processing of model outputs and combination with input data
- Statistical analysis and visualization

Due to privacy, licensing, and ethical constraints, no raw or processed data are included.
As a result, the code cannot be executed end-to-end without access to the original datasets.
However, the notebooks with analysis code contain sufficient comments and output to clarify the methods used.

## Repository Structure

- `data_preparation/`: Scripts for cleaning and preparing input data.
- `llm_prompting/`: Code for interacting with the LLM.
    - `prompts/`: Contains prompt templates used for classification and explanation generation.
    - `external_ressources/`: The external ressources *IHRA* and *Decoding Antisemitism Lexicon* (the latter needs to
      be downloaded from [here](https://link.springer.com/book/10.1007/978-3-031-49238-9) and stored as `lexicon.pdf` for the **LEXICON** prompt), the extracted *Taxonomy Examples* for the **STRUCT_EX** prompt, and a mapping between the Decoding annotation scheme and both external ressources. (Note that `STRUCT` and `STRUCT_EX` correspond to `TAX` and `TAX_EX` in the codebase.)
    - `classify_pdf_cache.py`: Code for classification prompting with the lexicon, which is computationally expensive and thus cached. 
    - `classify.py`: Code for classification prompting with all prompts except for the lexicon. 
- `output_processing/`: Code for processing and combining model outputs with input data. 
- `analyses/`: Notebooks for performing statistical analyses.
- `utils/`: Utility functions used across different parts of the codebase.
- `data/`: Datasets (not included).
    - `model_outputs/`: Model outputs (not included).
