import torch
import numpy as np
import random, time, datetime
from tqdm import tqdm
from pathlib import Path
from transformers import get_scheduler
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


from scibert.preprocessing.make_data import make

from scibert.config import (
    DATA,
    TEST_DIR,
    SEED,
    MODEL,
    BATCH_SIZE,
    EPOCHS,
    TOKENS_MAX_LENGTH,
    LEARNING_RATE,
    CKPTH_DIR,
)
from scibert.features.build_features import build_features
from scibert.models.dispatcher import MODELS


def initialize(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.derterministic = True
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    return device


def generate_inputs(X, y):
    input_ids = []
    attention_masks = []
    targets = []

    tokenizer = MODELS[MODEL]["tokenizer"]

    for i in tqdm(range(len(X))):
        X[i] = X[i].strip()

        inputs = tokenizer.encode_plus(
            X[i],
            add_special_tokens=True,
            max_length=TOKENS_MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids.append(inputs["input_ids"].squeeze(0))
        attention_masks.append(inputs["attention_mask"].squeeze(0))

        target = torch.zeros(2)
        target[y[i]] = 1
        targets.append(target)

    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    targets = torch.stack(targets, dim=0)

    return input_ids, attention_masks, targets


# define format_time function
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# define flat_accuracy function
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def evaluation_pipeline(model, testdataloader, device, loss_fn):
    ## MODEL EVALUATION

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for _, batch in enumerate(testdataloader):
            input_ids = batch[0]
            attention_masks = batch[1]
            labels = batch[2]

            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            logits = model(
                input_ids,  # set a sigmoid layer
                attention_masks,
            )

            loss = loss_fn(logits, labels)
            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = labels.to("cpu").numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

            predictions.extend(np.argmax(logits, axis=1).flatten().tolist())
            true_labels.extend(np.argmax(label_ids, axis=1).flatten().tolist())

    return total_eval_loss, total_eval_accuracy, true_labels, predictions


def training_pipeline(train_X, train_y, test_X, test_y, device):
    trainInput, trainMask, trainLabel = generate_inputs(train_X, train_y)
    traindataset = TensorDataset(trainInput, trainMask, trainLabel)

    testInput, testMask, testLabel = generate_inputs(test_X, test_y)
    testdataset = TensorDataset(testInput, testMask, testLabel)

    traindataloader = DataLoader(
        traindataset,
        sampler=RandomSampler(traindataset),
        batch_size=BATCH_SIZE,
    )

    testdataloader = DataLoader(
        testdataset,
        sampler=RandomSampler(testdataset),
        batch_size=BATCH_SIZE,
    )

    model = MODELS[MODEL]["model"]
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    total_steps = len(traindataloader) * EPOCHS

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    training_stats = {}
    # start training clock
    start_time = time.time()

    model.train()

    model.zero_grad()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        total_train_loss = 0

        for step, batch in enumerate(traindataloader):
            input_ids = batch[0]
            attention_masks = batch[1]
            labels = batch[2]

            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            logits = model(
                input_ids,
                attention_masks,
            )

            loss = loss_fn(logits, labels)

            total_train_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

            if step % 1 == 0:
                print(f"Loss at step {step}: {loss.item()}")

        torch.save(model.state_dict(), Path(CKPTH_DIR, f"{MODEL}_{epoch}.pt"))

        ## EVALUATION PIPELINE
        (
            total_eval_loss,
            total_eval_accuracy,
            true_labels,
            predictions,
        ) = evaluation_pipeline(model, testdataloader, device, loss_fn)

        training_stats[epoch] = {
            "Training Loss Per Epoch": total_train_loss,
            "Average Training Loss": total_train_loss / len(traindataloader),
            "Testing Loss Per Epoch": total_eval_loss,
            "Average Test Loss PE": total_eval_loss / len(testdataloader),
            "Average Test Accuracy PE": total_eval_accuracy / len(testdataloader),
            "Training Time": format_time(time.time() - start_time),
            "F1 Score": f1_score(true_labels, predictions),
            "Confusion Matrix": confusion_matrix(true_labels, predictions),
            "Accuracy (Sklearn)": accuracy_score(true_labels, predictions),
        }
        print(training_stats)

    print("Training complete!")


def main():
    device = initialize(SEED)

    traindf, testdf = make(DATA, TEST_DIR)

    train_X, train_y, test_X, test_y = build_features(traindf[:-1], testdf[:-1])

    training_pipeline(train_X, train_y, test_X, test_y, device)


if __name__ == "__main__":
    main()
