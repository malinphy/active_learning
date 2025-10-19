from utils.clean_text import clean_tweet
from utils.tweet_dataset import TweetDataset
from model.classifier_model import ClassifierModel
from sampling_methods.samplings import least_confidence_sampling, margin_sampling, entropy_sampling, sampling_function, active_learning_indices
from utils.training import train_one_epoch
from utils.prediction import evaluate
from utils.run_training import training_loop
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
from torch import optim
from torch.utils.data import DataLoader

if "__main__" == __name__:
    EPOCHS = 40
    learning_rate = 1e-6
    model_name = "google-bert/bert-base-uncased"
    BATCH_SIZE = 32
    random_state = 42
    torch.manual_seed(random_state)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    active_learning_function = least_confidence_sampling
    
    data_dir = r"data\hatespeech\labeled_data.csv"

    df = pd.read_csv(data_dir, 
                 usecols = ['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet'])

    df['first_index'] = np.arange(len(df))

    train_df, test_df = train_test_split(df, train_size = 0.8, random_state = random_state)

    balanced_df = train_df.groupby('class').apply(lambda x: x.sample(n=400, random_state=42)).reset_index(drop=True)

    training_indices = list(balanced_df['first_index'])

    remaining_train_df = train_df[~train_df['first_index'].isin(training_indices)].copy()


    
    train_dataset = TweetDataset(train_df)
    test_dataset = TweetDataset(test_df)
    balanced_dataset = TweetDataset(balanced_df)
    remaining_train_dataset = TweetDataset(remaining_train_df)


    train_dataloader = DataLoader(train_dataset, batch_size  = BATCH_SIZE, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE)
    balanced_dataloader = DataLoader(balanced_dataset, batch_size = BATCH_SIZE, shuffle = True)
    remaining_train_dataloader = DataLoader(remaining_train_dataset, batch_size = BATCH_SIZE, shuffle = False) 

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = ClassifierModel(output_dim= 3, model_name = model_name).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)




    N_CYCLES = 5          
    QUERY_SIZE = 100        

    all_cycle_metrics = {} 

    for cycle in range(N_CYCLES):
        print(f"\n🌀 ===== Active Learning Cycle {cycle + 1}/{N_CYCLES} =====")

       
        metrics = training_loop(
            model=model,
            train_loader=balanced_dataloader,
            test_loader=test_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            tokenizer=tokenizer,
            device=DEVICE,
            epochs=EPOCHS
        )

        all_cycle_metrics[f"cycle_{cycle+1}_train_losses"] = metrics["train_losses"]
        all_cycle_metrics[f"cycle_{cycle+1}_test_losses"] = metrics["test_losses"]
        all_cycle_metrics[f"cycle_{cycle+1}_train_accs"] = metrics["train_accs"]
        all_cycle_metrics[f"cycle_{cycle+1}_test_accs"] = metrics["test_accs"]
        all_cycle_metrics[f"cycle_{cycle+1}_train_f1s"] = metrics["train_f1s"]
        all_cycle_metrics[f"cycle_{cycle+1}_test_f1s"] = metrics["test_f1s"]

        test_loss, test_acc, test_f1_macro, _, cm, test_probs, index_probs = evaluate(
            model, remaining_train_dataloader, criterion, tokenizer, DEVICE
        )

       
        new_metrics = {"test_index_probabilites_per_epoch": index_probs}
        
        least_confident_indices = active_learning_indices(metrics, sampling_function , k=QUERY_SIZE)
       
        transport_df = remaining_train_df[remaining_train_df['first_index'].isin(least_confident_indices)]
        balanced_df = pd.concat([balanced_df, transport_df], ignore_index=True)
        remaining_train_df = remaining_train_df[~remaining_train_df['first_index'].isin(least_confident_indices)].copy()

        print(f"✅ Added {len(transport_df)} new samples to training set.")
        print(f"Remaining pool size: {len(remaining_train_df)}")

  
        balanced_dataset = TweetDataset(balanced_df)
        remaining_train_dataset = TweetDataset(remaining_train_df)

        balanced_dataloader = DataLoader(balanced_dataset, batch_size=BATCH_SIZE, shuffle=True)
        remaining_train_dataloader = DataLoader(remaining_train_dataset, batch_size=BATCH_SIZE, shuffle=False)

      
        model = ClassifierModel(output_dim=3, model_name=model_name).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

    print("\n🎯 Active Learning process completed!")

    # ✅ Save metrics for all cycles
    pd.to_pickle(all_cycle_metrics, "active_learning_metrics.pkl")
    print("📁 Metrics saved to 'active_learning_metrics.pkl'")
