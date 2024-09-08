# Fake News Detection Using Python and Machine Learning

## Project Overview
The **Fake News Detection** project is a machine learning solution designed to classify news articles as either **real** or **fake**. With the proliferation of online content, distinguishing between authentic and misleading information has become crucial. This project leverages **natural language processing (NLP)** techniques and machine learning algorithms to identify deceptive news articles.

## Features
- Classification of news articles as **real** or **fake**.
- Utilizes **TF-IDF Vectorization** for feature extraction from text data.
- Implements the **Passive-Aggressive Classifier**, a fast and efficient algorithm for binary classification problems.
- Achieves over **92% accuracy** in fake news detection.

## Technology Stack
- **Programming Language**: Python
- **Libraries**: 
  - `pandas` for data manipulation
  - `scikit-learn` for machine learning models
  - `NumPy` for numerical operations
  - `TfidfVectorizer` for text processing

## Dataset
The dataset contains **7796** news articles, with the following structure:
- **ID**: Unique identifier for each news article.
- **Title**: The title of the news article.
- **Text**: The main content of the article.
- **Label**: The classification label (REAL or FAKE).

The dataset is available for download [here](https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view).

## Approach
1. **Data Preprocessing**: 
   - The text data is cleaned and transformed using **TF-IDF Vectorization** to convert it into a numerical format suitable for machine learning models.
  
2. **Modeling**:
   - The **Passive-Aggressive Classifier** is used to train on the dataset. This algorithm adjusts the decision boundary for each misclassified sample while remaining passive for correct classifications, making it ideal for online learning and binary classification.

3. **Evaluation**:
   - The model is evaluated using **accuracy** and a **confusion matrix**, which shows the number of true positives, true negatives, false positives, and false negatives.

## Results
- The model achieves an accuracy of **92.82%**, demonstrating its effectiveness in detecting fake news.
- The confusion matrix shows:
  - **589** true positives (correctly identified real news)
  - **587** true negatives (correctly identified fake news)
  - **42** false positives (incorrectly identified real news as fake)
  - **49** false negatives (incorrectly identified fake news as real)

## Usage
### Prerequisites
To run this project, you'll need:
- Python 3 installed
- The following Python libraries installed:
  - `pandas`
  - `scikit-learn`
  - `numpy`
  
You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
