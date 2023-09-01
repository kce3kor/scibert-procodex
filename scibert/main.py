import torch
import random, time
from tqdm import tqdm
from pathlib import Path
from transformers import get_scheduler
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

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


def train(train_X, train_y, test_X, test_y):
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

        total_loss = 0

        for step, batch in enumerate(traindataloader):
            input_ids = batch[0]
            attention_masks = batch[1]
            labels = batch[2]

            logits = model(
                input_ids,
                attention_masks,
            )

            loss = loss_fn(logits, labels)

            total_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

            if step % 1 == 0:
                print(f"Loss at step {step}: {loss.item()}")

    avg_train_loss = total_loss / len(traindataloader)
    print(f"Average training loss: {avg_train_loss}")

    torch.save(model.state_dict(), Path(CKPTH_DIR, f"{MODEL}_{epoch}.pt"))


if __name__ == "__main__":
    initialize(SEED)

    traindf, testdf = make(DATA, TEST_DIR)

    train_X, train_y, test_X, test_y = build_features(traindf[:20], testdf[:20])

    train(train_X, train_y, test_X, test_y)
