import torch


class CustomBERTModel(torch.nn.Module):
    def __init__(self, model=None, hidden_dim=768, hidden_units=[64, 32, 16]):
        super(CustomBERTModel, self).__init__()

        self.model = model

        ## Types of Activation Functions: nn.GELU, nn.SiLU, nn.Mish

        # self.all_layers = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_dim, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.3),
        #     torch.nn.Linear(64, 32),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.3),
        #     torch.nn.Linear(32, 2),
        #     torch.nn.Sigmoid(),
        #     torch.nn.Softmax(dim=1),
        # )

        all_layers = []
        for hidden_unit in hidden_units:
            layer = torch.nn.Linear(hidden_dim, hidden_unit)
            all_layers.append(layer)
            # Put batch norm before activation function
            # But putting it after activation function is also fine
            # all_layers.append(torch.nn.BatchNorm1d(hidden_unit))
            all_layers.append(torch.nn.ReLU())

            # Put dropout after activation
            # all_layers.append(torch.nn.Dropout(0.5))

            hidden_dim = hidden_unit

        all_layers.append(torch.nn.Linear(hidden_units[-1], 2))
        all_layers.append(torch.nn.Sigmoid())
        # all_layers.append(torch.nn.Softmax(dim=1))

        self.all_layers = torch.nn.Sequential(*all_layers)

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
