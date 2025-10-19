from training import train_one_epoch
from prediction import evaluate

def training_loop(model, train_loader, test_loader, optimizer, criterion, tokenizer, device, epochs):
    # Track metrics
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    train_f1s, test_f1s = [], []
    train_probabilities_per_epoch = [] # To store probabilities from each epoch
    test_probabilities_per_epoch = []
    index_probs = []
    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")

        # ---- Training ----
        # CORRECTED LINE: Unpack all 5 returned values from train_one_epoch
        train_loss, train_acc, train_f1_macro, train_f1_micro, train_probs = train_one_epoch(
            model, train_loader, optimizer, criterion, tokenizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(train_f1_macro)
        train_probabilities_per_epoch.append(train_probs) # Store probabilities

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1_macro: {train_f1_macro:.4f}")

        # ---- Evaluation ----
        test_loss, test_acc, test_f1_macro, _, cm, test_probs, index_probs = evaluate(
            model, test_loader, criterion, tokenizer, device
        )
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        test_f1s.append(test_f1_macro)
        test_probabilities_per_epoch.append(test_probs)
        index_probs.append(index_probs)
        # print(f"\nConfusion Matrix:\n{cm}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1_macro: {test_f1_macro:.4f}")
        # print("=" * 60)


    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "train_f1s": train_f1s,
        "test_f1s": test_f1s,
        "train_probabilities_per_epoch": train_probabilities_per_epoch,
        "test_probabilities_per_epoch": test_probabilities_per_epoch,
        "test_index_probabilites_per_epoch" : index_probs
    }