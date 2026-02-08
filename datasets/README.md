# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Follow the download instructions below.

## Dataset 1: WikiText-2 (raw)

### Overview
- **Source**: HuggingFace `Salesforce/wikitext` / `wikitext-2-raw-v1`
- **Size**: train (36,718), validation (3,760), test (4,358) examples
- **Format**: HuggingFace Dataset
- **Task**: Language modeling / prompt corpus for activation analysis
- **Splits**: train, validation, test
- **License**: CC BY-SA 4.0 (see dataset card)

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset.save_to_disk("datasets/wikitext_2_raw_v1")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/wikitext_2_raw_v1")
```

### Sample Data
See `datasets/wikitext_2_raw_v1/samples.json` for 10 example records.

### Notes
- This dataset is small enough to use for quick activation probing.
- For larger-scale runs, consider WikiText-103 or The Pile (streaming).

## Dataset 2: LAMBADA

### Overview
- **Source**: HuggingFace `cimec/lambada`
- **Size**: train (2,662 novels), dev (4,869 passages), test (5,153 passages)
- **Format**: HuggingFace Dataset
- **Task**: Long-range dependency word prediction
- **Splits**: train, validation/dev, test
- **License**: CC BY 4.0 (see dataset card)

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

dataset = load_dataset("lambada")
# or use the dataset card namespace
# dataset = load_dataset("cimec/lambada")

dataset.save_to_disk("datasets/lambada_full")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/lambada_full")
```

### Sample Data
See `datasets/lambada_train_1k/samples.json` for 10 example records
from the 1k-sample subset stored in this workspace.

### Notes
- Good for testing whether concept features require broader context.
- The full dataset is moderate in size; the workspace currently stores a 1k subset.

## Other Recommended Datasets (not downloaded)

### CounterFact
- **Source**: ROME project (CounterFact dataset)
- **Use**: Knowledge editing / causal tracing evaluations
- **Download**: See `code/rome/` for dataset utilities and links in README.

### The Pile / C4
- **Use**: Large-scale pretraining corpus for collecting activation samples
- **Download**: Use HuggingFace streaming or author-hosted buckets

