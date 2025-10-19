import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def train_one_epoch(model, dataloader, optimizer, criterion, tokenizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    all_probabilities = [] # To store probabilities for least confidence sampling

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        tweets = list(batch["tweets"])
        labels = batch["labels"].long().to(device)

        tweet_tokens = tokenizer(
            tweets,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        ).to(device)

        optimizer.zero_grad()
        preds, _ = model(tweet_tokens)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(preds, dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(torch.softmax(preds, dim=1).detach().cpu().numpy())

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_micro = f1_score(all_labels, all_preds, average="micro")

    return avg_loss, acc, f1_macro, f1_micro, all_probabilities


