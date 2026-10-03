# Phishing Email Detection Model

A simple machine learning project that detects whether an email is likely to be **Phishing** or **Safe**.

The project uses email text and URL-related features to train a machine learning model using Python and Scikit-learn.

## Features

- Detects phishing and safe emails
- Analyzes email text using TF-IDF
- Extracts URL-related features
- Uses Logistic Regression for classification
- Displays model accuracy
- Generates a confusion matrix
- Allows users to test new email messages

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy
- Hugging Face Datasets

## How It Works

The email content is first converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The number of URLs present in an email is also extracted as an additional feature.

These features are combined and provided to a **Logistic Regression** model.

The model then classifies the email as:

- **Safe**
- **Phishing**

## Dataset

This project uses the **Seven Phishing/Spam Email Datasets** available on Hugging Face.

Dataset source:

https://huggingface.co/datasets/puyang2025/seven-phishing-email-datasets

The dataset contains email content and labels that are used for training and testing the model.

For more information about the dataset, refer to the dataset documentation.

## Model Evaluation

The model provides:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The actual results are generated when the model is trained and tested.

## Project Structure

```text
Phishing-Email-Detection/
│
├── dataset/
│   └── README.md
│
├── screenshots/
│   ├── phishing-email.png
│   ├── safe-email.png
│   └── confusion-matrix.png
│
├── phishing_detector.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

How to Run

Install the required Python libraries:
pip install -r requirements.txt

Run the program:
python phishing_detector.py
The program will load the dataset, train the model, display the evaluation results, and allow you to test an email.

Screenshots
Screenshots of the project results will be added here after testing the model.

Limitations
This project is developed for educational and internship purposes.
Phishing techniques continuously change, so a model trained on historical email data may not detect every modern phishing attempt.
The model should not be considered a replacement for a production-grade email security system.

Future Improvements
Add email header analysis
Improve URL feature extraction
Add sender and domain reputation checks
Compare multiple machine learning algorithms
Add a web-based interface
Improve detection of newly emerging phishing techniques

Author
Bharath M
Cybersecurity Engineering Student

GitHub:
https://github.com/bharath-1206⁠�
