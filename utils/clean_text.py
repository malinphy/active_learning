import re

def clean_tweet(text: str) -> str:
    # Remove RT @username and following characters up to the first space
    text = re.sub(r'RT\s*@\S+\s*', '', text)
    # Remove any remaining @mentions (like inside the text)
    text = re.sub(r'@\S+', '', text)
    # Remove punctuation
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Normalize spaces and lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text