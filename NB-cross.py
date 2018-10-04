import pandas as pd
from os.path import join
from lxml import etree as ET
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

tree = ET.parse(join("data", "rest_final.xml"))
root = tree.getroot()

reviews = root.findall("Review")
sentences = root.findall("**/sentence")
print("# Reviews   : ", len(reviews))
print("# Sentences : ", len(sentences))

datas = []
categories = []
for i in root.iter('sentence'):
    if(i.get('OutOfScope') != 'TRUE'):
        opinion = i.find('Opinion')
        if(opinion.attrib['category'] == 'REST#QUALITY'):
            text = i.find('text')
            datas.append(text.text)
            categories.append(opinion.attrib['category'])
        else:
            text = i.find('text')
            text = text.text
            datas.append(text)
            categories.append('None')

df = pd.DataFrame({"datas": datas, "categories": categories})
scoring = 'accuracy'
X_train, X_valid, y_train, y_valid = train_test_split(df['datas'], df['categories'], test_size=0.2, random_state=42, shuffle= True)
print(X_train.shape)
print(y_train.shape)
vectorizer = CountVectorizer()
transformed_x_train = vectorizer.fit_transform(X_train).toarray()
trainVocab = vectorizer.vocabulary_
vectorizer = CountVectorizer(vocabulary=trainVocab)
transformed_x_valid = vectorizer.fit_transform(X_valid).toarray()
model = MultinomialNB()
kfold = model_selection.KFold(n_splits=10, random_state=50)
cv_results = model_selection.cross_val_score(model, transformed_x_train, y_train, cv=kfold, scoring=scoring)
msg = "Result: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

clf1 = SVC(kernel = 'linear')
clf1.fit(transformed_x_train, y_train)
y_pred = clf1.predict(transformed_x_valid)
print(accuracy_score(y_valid, y_pred))
print(confusion_matrix(y_valid, y_pred))
print(classification_report(y_valid, y_pred))