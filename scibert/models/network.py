import torch


class CustomBERTModel(torch.nn.Module):
    def __init__(self, model, hidden_dim):
        super(CustomBERTModel, self).__init__()

        self.model = model
        self.hidden_dim = hidden_dim

        # nn.GELU, nn.SiLU, nn.Mish
        self.all_layers = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 2),
            torch.nn.Sigmoid(),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, input_ids, attention_masks):
        outputs = self.model(
            input_ids,
            token_type_ids=None,
            attention_mask=attention_masks,
        )
        cls = outputs[0][:, 0, :]

        x = self.all_layers(cls)

        return x


if __name__ == "__main__":
    model = CustomBERTModel("allenai/scibert_scivocab_uncased")

    print(model)
