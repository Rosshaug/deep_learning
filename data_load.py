from datasets import load_dataset
import random

# --- Configuration ---

# datasets==3.6.0

SAMPLES_EN = 1000  # Target for English (approx. 50% of data)

DATASET_NAME = "cc100" # The generic dataset name


def sample_and_save(lang_code, num_samples, output_file):
    print(f"Sampling {num_samples} records for {lang_code}...")

    # Using the generic CC-100 structure as a robust example:
    streamed_ds = load_dataset(DATASET_NAME, lang_code, split='train', streaming=True, trust_remote_code=True)

    sampled_ds = streamed_ds.take(num_samples)

    data_list = list(sampled_ds)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in data_list:
            text = record['text'].strip()
            if text:
                f.write(text + '\n')

    print(f"Saved {len(data_list)} records to {output_file}")

sample_and_save('en', SAMPLES_EN, './data/cc100_en_subset.txt')

print("All language subsets have been successfully sampled and saved.")