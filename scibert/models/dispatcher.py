from transformers import AutoModel, AutoTokenizer

from scibert.models.network import CustomBERTModel

MODELS = {
    "scibert": {
        "model": CustomBERTModel(
            model=AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", output_hidden_states=True),
            hidden_dim=128,
        ),
        "tokenizer": AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased"),
    },
    "tinyBERT": {
        "model": CustomBERTModel(
            model=AutoModel.from_pretrained("prajjwal1/bert-tiny", output_hidden_states=True),
            hidden_dim=128,
            hidden_units=[128, 64, 32],
        ),
        "tokenizer": AutoTokenizer.from_pretrained("prajjwal1/bert-tiny"),
    },
}


if __name__ == "__main__":
    pass
