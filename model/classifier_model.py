from transformers import AutoModel, AutoConfig
import torch.nn as nn
import torch.nn.functional as F

class ClassifierModel(nn.Module):
    def __init__(self,output_dim, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = AutoConfig.from_pretrained(model_name).hidden_size
        self.fc_layer_1 = nn.Linear(self.hidden_size, output_dim)

    def vectorizer(self, output):
        return output.last_hidden_state[:, 0, :]

    def forward(self, tweet_tokens):
        model_output = self.model(tweet_tokens['input_ids'], attention_mask=tweet_tokens['attention_mask'])
        vectors = self.vectorizer(model_output)
        vectors = F.normalize(vectors, p=2, dim=1)
        return self.fc_layer_1(vectors),vectors