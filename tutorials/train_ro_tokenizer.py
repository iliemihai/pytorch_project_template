from pathlib import Path

from tokenizers import BertWordPieceTokenizer


# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer()

# Customize training
tokenizer.train(files="/home/mihai/Downloads/ro_dedup.txt", vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save(".", "roberto")