# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Follow the download instructions below.

## Dataset 1: WikiText-2 (raw)

### Overview
- **Source**: HuggingFace `wikitext` / `wikitext-2-raw-v1`
- **Size**: train (36,718), validation (3,760), test (4,358) examples
- **Format**: HuggingFace Dataset
- **Task**: Language modeling / prompt corpus for activation analysis
- **Splits**: train, validation, test
- **License**: See HuggingFace dataset card

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset.save_to_disk("datasets/wikitext_2_raw")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/wikitext_2_raw")
```

### Sample Data
See `datasets/wikitext_2_raw/samples/samples.json` for 10 example records.

### Notes
- This dataset is small enough to use for quick activation probing.
- For larger-scale runs, consider WikiText-103 or The Pile (streaming).

## Dataset 2: CHIP (Compositionality of Human-annotated Idiomatic Phrases)

### Overview
- **Source**: Liu & Neubig (2022) dataset released with their compositionality paper
- **Task**: Phrase compositionality / idiom vs. literal pairs
- **Use case**: Evaluate compound concept representation and compositionality

### Download Instructions

**From the authors' repo (recommended):**
1. Visit the paper's code repository (linked in the paper).
2. Download the `CHIP` dataset files and place them under:
   `datasets/chip/`

### Loading the Dataset
- The dataset is distributed as text/CSV/JSON files in the repo; load with pandas or json as appropriate.

### Notes
- This dataset is small and ideal for targeted compositionality tests.
- If the repo structure changes, search for `CHIP` in the repository.
