import torch


class CustomBERTModel(torch.nn.Module):
    def __init__(self, model):
        super(CustomBERTModel, self).__init__()

        self.model = model
        # set a linear layer to map the hidden states to the output space
        self.linear = torch.nn.Linear(768, 2)
        # set a dropout layer
        self.dropout = torch.nn.Dropout(0.3)
        # set a relu activation function
        self.relu = torch.nn.ReLU()

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_masks):
        # pass the input to the model

        outputs = self.model(
            input_ids,
            token_type_ids=None,
            attention_mask=attention_masks,
        )

        # pass the last hidden state of the token `[CLS]` to the linear layer
        x = self.linear(outputs[0][:, 0, :])

        # pass the output of the linear layer to the relu activation function
        x = self.relu(x)

        # pass the output of the relu activation function to the dropout layer
        x = self.dropout(x)

        # set a sigmoid layer
        x = torch.sigmoid(x)

        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = CustomBERTModel("allenai/scibert_scivocab_uncased")

    print(model)
