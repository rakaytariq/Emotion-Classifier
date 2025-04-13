from src.data_loader import load_and_tokenize_dataset
from src.trainer import train_model

def main():
    model_name = "distilbert-base-uncased"
    dataset, tokenizer = load_and_tokenize_dataset(model_name)
    train_model(model_name, dataset, tokenizer)

if __name__ == "__main__":
    main()
