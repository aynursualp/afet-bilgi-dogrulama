🚨 Disaster Misinformation Detection (NLP) 🚨

A machine learning prototype to detect fake news and disinformation during disaster periods.

💡 Why this project? During emergencies and disasters, the rapid spread of unverified information can hinder rescue operations. This tool uses Natural Language Processing (NLP) to classify incoming social media messages as either verified official information or disinformation.

🚀 Features

🧩 Object-Oriented Architecture: Clean, modular (OOP), and scalable codebase.

🧹 Smart Text Preprocessing: Automatically removes punctuation, numbers, and stop-words for accurate analysis.

🧠 Machine Learning Engine: Powered by the Logistic Regression algorithm via scikit-learn.

📂 External Data Management: Seamlessly reads and processes training datasets from structured JSON files.

🛠️ Technologies Used

Language: Python 

Libraries: scikit-learn, json, string

Data Format: JSON

📋 Installation

To run this project on your local machine, follow these simple steps:

1. Clone the repository:
```bash
git clone https://github.com/aynursualp/afet-bilgi-dogrulama.git
```

2. Navigate to the project directory:
```bash
cd disaster-check
```

3. Install the required libraries:
```bash
pip install -r requirements.txt
```

💻 Usage

Simply run the main Python file to train the model and test it with a sample message:

```bash
python main.py
```
