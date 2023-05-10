import torch
from transformers import AutoTokenizer, MBartForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
model = MBartForSequenceClassification.from_pretrained("facebook/mbart-large-50", num_labels=2)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits


print("logits",logits)

predicted_class_id = logits.argmax().item()

print("pred_cl", predicted_class_id)



# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
#num_labels = len(model.config.id2label)
#model = MBartForSequenceClassification.from_pretrained("facebook/mbart-large-cc25", num_labels=num_labels)

#inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")


