from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class EmotionClassifier:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.labels = [
            'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
        ]

    def predict_emotion(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            top_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][top_class].item()

        return self.labels[top_class], confidence

# Example usage (to test locally):
if __name__ == "__main__":
    classifier = EmotionClassifier()
    text = "I feel very alone and sad today."
    emotion, confidence = classifier.predict_emotion(text)
    print(f"Detected Emotion: {emotion} (Confidence: {confidence:.2f})")
