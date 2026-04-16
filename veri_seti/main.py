import json
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class MisinformationDetector:
    def __init__(self):
        # Model's core settings and stop-words definition
        self.stop_words = ["bu", "ve", "ile", "için", "bir", "çok", "lütfen", "herkes"]
        self.vectorizer = CountVectorizer()
        self.model = LogisticRegression()
        self.is_trained = False

    def clean_text(self, text):
        text = text.lower()
        text = "".join([char for char in text if not char.isdigit()])
        text = "".join([char for char in text if char not in string.punctuation])
        
        words = text.split()
        meaningful_words = [word for word in words if word not in self.stop_words]
        
        return " ".join(meaningful_words) 

    def load_json_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def train_model(self, file_path):
        # Load data and split into lists
        dataset = self.load_json_data(file_path)
        
        clean_data = []
        labels = []

        for data in dataset:
            clean_sentence = self.clean_text(data["metin"])
            clean_data.append(clean_sentence)
            labels.append(data["etiket"])

        # train the model
        vectors = self.vectorizer.fit_transform(clean_data)
        self.model.fit(vectors, labels)
        self.is_trained = True
        print("✅ AI model successfully trained with JSON dataset!")

    def predict(self, message):
        if not self.is_trained:
            return "Error: The model is not trained yet!"
        
        clean_message = self.clean_text(message)
        new_vector = self.vectorizer.transform([clean_message])
        prediction = self.model.predict(new_vector)[0]
        
        if prediction == 1:
            return "⚠️ POTENTIAL DISINFORMATION (FAKE NEWS)."
        else:
            return "✅ APPEARS TO BE VERIFIED (OFFICIAL)."

if __name__ == "__main__":
    disaster_model = MisinformationDetector()
    
    disaster_model.train_model("veri_seti.json")
    
    test_message = "AFAD duyurusu: ŞOK İDDİA baraj patladı acil yayalım!!!"
    print(f"\nTested message: {test_message}")
    
    result = disaster_model.predict(test_message)
    print(f"AI DECISION: {result}")