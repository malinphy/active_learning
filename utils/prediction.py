import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score



def evaluate(model, dataloader, criterion, tokenizer, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    all_probabilities = []
    index_prob_pairs = []
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            tweets = list(batch["tweets"])
            labels = batch["labels"].long().to(device)
            first_indices = batch['indices']
            
            tweet_tokens = tokenizer(
                tweets,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128
            ).to(device)

            logits, _ = model(tweet_tokens)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()

            # Append (index, probability) pairs
            for idx, prob in zip(first_indices, probabilities):
                index_prob_pairs.append({"index": idx, "probabilities": prob})

            all_probabilities.extend(probabilities)
    
    avg_loss = total_loss / len(dataloader)
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_micro = f1_score(all_labels, all_preds, average="micro")

    return avg_loss, acc, f1_macro, f1_micro, cm, all_probabilities, index_prob_pairs