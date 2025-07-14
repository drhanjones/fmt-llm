The paper utilizes the 10M and 100M datasets from the [BabyLM Challenge 2023](https://babylm.github.io/archive_2023.html).

**Dataset Structure:**
- Each dataset should be placed in a separate folder.
- Each folder must contain:
    - `train.bin`
    - `val.bin`
    - `meta.pkl`
    - `tokenizer.json`

To generate these files, run or adapt the `prepare.py` script.

NOTE: The models in the paper have a vocabulary size of 8000.