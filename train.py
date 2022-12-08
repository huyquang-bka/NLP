# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle
from util import *


train_path = "data/full_train.csv"
model_save_path = "model.pkl"

train_data = pd.read_csv(train_path)
train_data = train_data.dropna()
# train_neg = train_data[train_data.Rating == 0.0]
# train_data_pos = train_data[train_data.Rating ==
#                             1.0].sample(n=len(train_neg), random_state=62)
# train_data = pd.concat([train_neg, train_data_pos])
print(train_data.head())
pos_data = []
pos_label = []
neg_data = []
neg_label = []


# # Thêm mẫu bằng cách lấy trong từ điển Sentiment (nag/pos)
for index, row in enumerate(pos_list):
    pos_data.append(row)
    pos_label.append(1)
for index, row in enumerate(nag_list):
    neg_data.append(row)
    neg_label.append(0)


############################
print("Create train/test data...")
X_train, X_test, y_train, y_test = train_test_split(
    train_data.Comment, train_data.Rating, test_size=0.2, random_state=62)
print("Start transform train data...")
X_train, y_train = transform_to_dataset(X_train, y_train)
X_train = X_train + pos_data + neg_data
y_train = y_train + pos_label + neg_label
print("Start transform test data...")
X_test, y_test = transform_to_dataset(X_test, y_test)


############################
classifier = LinearSVC(
    fit_intercept=True, multi_class='crammer_singer', C=1, verbose=1)
print("Create pipeline")
clf = Pipeline([('CountVectorizer', CountVectorizer(
    ngram_range=(1, 5), stop_words=stop_ws, max_df=0.5, min_df=5)),
    ('tfidf', TfidfTransformer(
        use_idf=False, sublinear_tf=True, norm='l2', smooth_idf=True)),
    ('classifier', classifier)])
############################

print("Start training...")
clf.fit(X_train, y_train)
with open(model_save_path, 'wb') as f:
    pickle.dump(clf, f)
print("Training done!")
print("Start testing...")
score = clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
print(score)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(clf, X_test, y_test)
plt.savefig('confusion_matrix.png')
plt.show()
print("Test done!")
