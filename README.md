## tl;dr
* I worked with a company to automatically convert unstructured text into structured data by using a linear chain conditional random fields (chain CRF) model.
* Performance of chain CRF model: overall 87% accuracy, but poor precision for unbalanced minority classes
* Next steps: train and improve model on a larger and more representative dataset
* Tools utilized: Python, Flask, mongodb

## Motivation
I partnered with a wellness company that provides health management programs, which include personalized coaching and monitoring of customers' health & behavioral data, such as nutrition, fitness, sleep, and weight. As this company is growing, their customer base is increasing, and each customer is generating more data on a daily basis. However, a lot of this data, such as food logs where customers record information about their nutritional intake, exists in an unstructured format, making it difficult to extract potentially valuable insights out of this data.

My goal was to apply a machine learning model that can transform unstructured food logs into structured format, so that the company can do more efficient analysis and understand customer behavior at a more granular level.

## Data Preparation / Labelling Training Data
In order to build a supervised model, I manually labeled around 350 food logs from one customer, which contained around 3000 total words. The labels I used were:
1. meal
2. time
3. quantity
4. unit
5. food
6. drink
7. comment
8. other

I built a web app using Flask framework and mongodb to make the labelling process more efficient. Also, the web app allows for a more scalable approach to to be taken when labelling and/or training a model on newly generated, unseen data.  


## Feature Engineering
I engineered the following features for each word / token in a food log:
* spaCy part of speech (POS) tags
* boolean for containing a number
* boolean for containing punctuation, such as a slash or dash
* POS tags for the words before and after current word

## Modeling
I used a linear chain conditional random field (chain CRF) model to predict the labels for each word. Chain CRF is a discriminative graphical model which predicts sequences of labels for sequences of input samples. In other words, chain CRF models a conditional distribution (the probability that a word belongs to a certain class/label given its features), and it does this through a graphical structure that stores the features as nodes and connects the features with edges.


For my baseline model, I used Naive Bayes, which is oftentimes used for text classification purposes, and is generally known to perform pretty well, especially on small training datasets. However, Naive Bayes is not able to take into consideration sequential ordering of words, while chain CRF can, due to its inherent graphical nature.


## Results
I validated my model with 5-fold cross validation, and overall the Chain CRF model returned an accuracy of ~87%. Accuracy is defined as the number of correctly predicted labels out of the total number of predicted labels. In terms of per-class scoring, the model predicted 'food', 'meal', 'quantity', 'comment', and 'other' labels well, at above 85% precision and recall. It performed moderately well on 'time' labels (68% f1-score), and performed poorly on 'unit' and 'drink' labels, which were minority classes.  


## Next Steps
This project demonstrated a proof of concept for using linear chain CRF to predict the labels of unstructured customer-generated text data. Due to data security reasons, I was only able to demonstrate the performance of the model with a small dataset. As next steps, the accuracy of the model can be improved given a larger and more representative training set. If implemented, this model could allow the company to perform much deeper analysis on customers' data. For example, with structured data, the company can identify patterns in nutritional behavior that are contributing to health outcomes, build predictive models to assess customers' health risks, and target interventions to customers who are at higher risk.


## Technologies Used
* python (numpy, pandas, pystruct, scikit-learn, json)
* spaCy, nltk
* git
* mongodb, pymongo
* Flask, HTML/CSS
* matplotlib, seaborn
