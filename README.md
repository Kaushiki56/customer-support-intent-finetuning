# Customer Support Intent Classification Using DistilBERT

## 1. Project Overview

This project implements a customer support intent-classification system using supervised fine-tuning of DistilBERT.

The objective is to automatically classify customer messages into eight categories, helping organize support requests and route them to the appropriate department.

## 2. Problem Statement

Customer support teams receive messages about payment failures, refunds, delivery delays, account access, cancellations, product inquiries, returns, and complaints.

Manually categorizing these messages can be time-consuming. This project investigates whether fine-tuning a pretrained transformer model can classify these intents automatically.

## 3. Technology Stack

- Python
- Google Colab
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- DistilBERT
- Pandas and NumPy
- Scikit-learn
- Matplotlib

## 4. Dataset

The project uses a synthetically generated customer support dataset.

The final dataset contains 688 unique messages across eight intent classes.

| Label | Intent |
|---|---|
| 0 | Payment Issue |
| 1 | Refund Request |
| 2 | Delivery Issue |
| 3 | Account Access |
| 4 | Order Cancellation |
| 5 | Product Inquiry |
| 6 | Return / Exchange |
| 7 | General Complaint |

The dataset is cleaned by checking missing values, normalizing message text, and removing exact duplicate messages.

The dataset is divided into training, validation, and test sets. The test set contains 104 examples.

Because the dataset is synthetic, the reported results should not be interpreted as guaranteed performance on real customer support conversations.

## 5. Model and Training

The pretrained `distilbert-base-uncased` model is used for eight-class text classification.

The tokenizer converts customer messages into token IDs and attention masks. A classification head predicts one of the eight intents.

The model is fine-tuned using supervised learning and cross-entropy loss.

Training configuration:

- Maximum sequence length: 128 tokens
- Training batch size: 8
- Evaluation batch size: 8
- Learning rate: 2e-5
- Number of epochs: 4
- Weight decay: 0.01
- Random seed: 42

## 6. Evaluation

The fine-tuned model achieved the following results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 98.08% |
| Weighted Precision | 98.46% |
| Weighted Recall | 98.08% |
| Weighted F1-score | 98.12% |

The model correctly classified 102 out of 104 test messages.

The confusion matrix and individual predictions are available in the `results/` directory.

## 7. Baseline Comparison

The pretrained DistilBERT backbone with a newly initialized classification head achieved 5.77% test accuracy before fine-tuning.

After supervised fine-tuning, test accuracy increased to 98.08%.

The baseline is an untrained classification head, not an already-trained customer intent classifier. The comparison demonstrates the effect of training the model for this classification task.

## 8. Dataset Quality and Limitations

The final dataset contains no missing text values, missing labels, or exact duplicate messages.

The class distribution is not perfectly balanced. Account Access contains 48 examples, and General Complaint contains 40 examples, while the other six classes contain 100 examples each.

The dataset is synthetically generated and may contain recurring templates. Similar templates across data splits could make test performance appear better than performance on independently collected messages.

The model also made two errors on the test set, demonstrating that high overall accuracy does not guarantee that every message will be classified correctly.

## 9. Repository Structure

- `Customer_Support_Intent_FineTuning.ipynb`: Main notebook containing data preparation, training, evaluation, and inference.
- `data/`: Final dataset and saved data splits.
- `results/`: Evaluation metrics, predictions, confusion matrix, and baseline comparison.
- `requirements.txt`: Python dependencies.

## 10. How to Reproduce

1. Open the notebook in Google Colab.
2. Install the dependencies listed in `requirements.txt`.
3. Load the dataset or run the dataset-creation cells.
4. Perform data cleaning and prepare the training, validation, and test splits.
5. Load the pretrained tokenizer and DistilBERT model.
6. Tokenize the datasets.
7. Run the supervised fine-tuning pipeline.
8. Evaluate the model on the held-out test set.
9. Generate the confusion matrix and baseline comparison.
10. Run the demonstration on new customer messages.

A GPU runtime is recommended for faster training.

For exact reproduction, use the saved dataset splits, model configuration, random seeds, and library versions from the original experiment.

## 11. Future Improvements

- Evaluate the model on independently collected real customer messages.
- Expand and diversify the training dataset.
- Investigate errors between similar intents.
- Compare DistilBERT against a TF-IDF and logistic regression baseline.
- Test whether improved deduplication and template-aware splitting affect generalization.

## 12. Disclaimer

This is an educational text-classification project. The model is intended to classify customer support messages and does not independently resolve customer complaints or perform financial transactions.
