from lxml import etree as ET
from os.path import join
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

tree = ET.parse(join("data", "rest_final_new.xml"))
root = tree.getroot()

reviews = root.findall("Review")
sentences = root.findall("**/sentence")
print("# Reviews   : ", len(reviews))
print("# Sentences : ", len(sentences))
count = 0
# count_1 = 0
datas = []
categories = []
for i in root.iter('sentence'):
    # count_1+=1
    # print("la: " + str(count_1))
    if(i.get('OutOfScope') != 'TRUE'):

        opinion = i.find('Opinion')

        if(opinion.attrib['category'] == 'REST#AMBIENCE'):
            text = i.find('text')
            datas.append(text.text)
            categories.append(opinion.attrib['category'])
        else:
            text = i.find('text')
            text = text.text
            datas.append(text)
            categories.append('None')
    else:
        count+=1
# print(datas)
# print(categories)
# print(count)
df = pd.DataFrame({"datas": datas, "categories": categories})
# print(datas)
# print(df)
X_train, X_valid, y_train, y_valid = train_test_split(datas, categories, test_size=0.2, random_state=50)
print(y_valid)
vectorizer = CountVectorizer()
transformed_x_train = vectorizer.fit_transform(X_train)
trainVocab = vectorizer.vocabulary_
vectorizer = CountVectorizer(vocabulary=trainVocab)
transformed_x_valid = vectorizer.fit_transform(X_valid)

best_clf = MultinomialNB()
best_clf.fit(transformed_x_train, y_train)
y_pred = best_clf.predict(transformed_x_valid)
# for i in range (0,len(y_pred)):
#     # print(y_pred[0])
#     if y_pred[i] != 'None':
#         y_pred_true.append(y_pred[i])
# print(y_pred_true)

# df_pred = pd.DataFrame({"categories": y_pred})
# print(df_pred['categories']!="None")
# df_valid = pd.DataFrame({"categories": y_valid})
# print(df_pred['categories']!="None")

print('Training size = %d, accuracy = %.2f%%' % \
      (len(X_train), accuracy_score(y_valid, y_pred) * 100))
# a = "Giá_cả hợp_lí , khá hài_lòng"
# test = vectorizer.fit_transform(a).toarray()
# a = best_clf.predict(test)
# print(a)


# # pred = []
# params = {'alpha': [0.38,0.39,0.40,0.41,0.42,0.43]}
# clf = MultinomialNB()
# clf = GridSearchCV(clf, params, cv=5)
# clf.fit(transformed_x_train, y_train)
#
# # print(clf.best_params_)
# best_clf = clf.best_estimator_
# y_pred = best_clf.predict(transformed_x_valid)
# df1 = pd.DataFrame({"datas": X_valid, "categories": y_pred})
# print(df1)
# print(clf.best_params_)
# print('Training size = %d, accuracy = %.2f%%' % \
#       (len(X_train), accuracy_score(y_valid, y_pred) * 100))

# clf1 = SVC(kernel = 'linear')
# clf1.fit(transformed_x_train, y_train)
# y_pred = clf1.predict(transformed_x_valid)
# print('Training size = %d, accuracy = %.2f%%' % \
#       (len(X_train), accuracy_score(y_valid, y_pred) * 100))
print(confusion_matrix(y_valid, y_pred))