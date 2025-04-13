from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize_dataset(model_name, dataset_name="dair-ai/emotion"):
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)
    
    encoded = dataset.map(tokenize, batched=True)
    
    return encoded, tokenizer 
