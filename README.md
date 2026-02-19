# 📧 Spam Email Detection using Machine Learning

## Overview

This project implements a Spam Email Detection system using:

- TF-IDF Vectorization
- Multinomial Naïve Bayes Classifier

It classifies emails into:
- Spam (1)
- Not Spam (0)

---

## 🧠 Mathematical Background

We model spam detection as a binary classification problem.

Using Bayes Theorem:

P(Spam | X) = P(X | Spam) * P(Spam) / P(X)

Multinomial Naïve Bayes assumes conditional independence between words:

P(X | Spam) = ∏ P(word_i | Spam)

TF-IDF weighting improves feature representation by:

TF-IDF = TF × log(N / DF)

Where:
- TF = Term Frequency
- DF = Document Frequency
- N = Total number of documents

---

## 📂 Project Structure

