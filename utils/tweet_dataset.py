from torch.utils.data import Dataset

class TweetDataset(Dataset):
    def __init__(self, df):
        self.df = df 
    
    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self, index):
        return {
            "tweets": self.df['tweet'].iloc[index], 
            "labels": self.df['class'].iloc[index], 
            "indices" : self.df['first_index'].iloc[index]
        }