import torch


class CustomBERTModel(torch.nn.Module):
    def __init__(self, model, hidden_dim):
        super(CustomBERTModel, self).__init__()

        self.model = model
        self.hidden_dim = hidden_dim
        # set a linear layer to map the hidden states to the output space
        self.linear1 = torch.nn.Linear(self.hidden_dim, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 2)
        # set a dropout layer
        self.dropout = torch.nn.Dropout(0.3)
        # set a relu activation function
        self.tanh = torch.nn.Tanh()

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_masks):
        # pass the input to the model

        outputs = self.model(
            input_ids,
            token_type_ids=None,
            attention_mask=attention_masks,
        )

        # pass the last hidden state of the token `[CLS]` to the linear layer
        x = self.linear1(outputs[0][:, 0, :])
        x = self.tanh(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.tanh(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = torch.sigmoid(x)

        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = CustomBERTModel("allenai/scibert_scivocab_uncased")

    print(model)
