import pickle

from lxml import etree as ET
from os.path import join
import pandas as pd
from sklearn import model_selection
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn_crfsuite import estimator


def SaveModel(clf):
    filename = 'PRICES_new.pkl'
    saved_model = open(join("models_new_1",filename), 'wb')
    pickle.dump(clf, saved_model)
    saved_model.close()

datas = []
categories = []
datas_valid = []
categories_valid = []

with open(join("data_train", "datas_PRICES_new.txt"),'r', encoding='utf-8')as file:
    for i in file:
        datas.append(i)

with open(join("data_train", "labels_PRICES_new.txt"),'r', encoding='utf-8')as file:
    for i in file:
        categories.append(i)


df = pd.DataFrame({"datas": datas, "categories": categories})
data = df['datas']
label = df['categories']

X_train = datas
y_train = label
# X_valid = datas_valid
# y_valid = categories_valid


text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)), ])

print(text_clf_svm.get_params().keys())
param_grid = {
    "clf__alpha": np.linspace(0.1, 1, 20)
    # "clf__gamma": np.linspace(0.1, 1, 4),
}
text_clf_svm = GridSearchCV(text_clf_svm, param_grid=param_grid, cv=5)
text_clf_svm = text_clf_svm.fit(X_train, y_train)
print(text_clf_svm)
# predicted_SVM = text_clf_svm.predict(X_valid)
# with open("predicted.txt",'w',encoding='utf-8') as f:
#     for i in predicted_SVM:
#         f.write(i)
    # print(np.mean(predicted_SVM == y_valid))
SaveModel(text_clf_svm)
# print('Training size = %d, accuracy = %.2f%%' % \
#       (len(X_train), accuracy_score(y_valid,  predicted_SVM ) * 100))
# # print(classification_report(y_valid, y_pred))
# print(confusion_matrix(y_valid, predicted_SVM ))

