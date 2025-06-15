from transformers import BertTokenizer, BertPreTrainedModel
import torch

# model_name = "bert-base-uncased"
model_name = "AbstractPhil/bert-beatrix-2048"



tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertPreTrainedModel.from_pretrained(model_name).eval()

def encode_labels(label_list):
    with torch.no_grad():
        inputs = tokenizer(label_list, return_tensors="pt", padding=True, truncation=True)
        outputs = bert(**inputs)
        # Use the last hidden state as the label embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings


if __name__ == "__main__":
    labels = ["cat", "dog", "fish"]
    embeddings = encode_labels(labels)
    print(embeddings.shape)  # Should be [3, 768]
    print(embeddings)