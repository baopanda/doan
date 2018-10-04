from lxml import etree as ET
from os.path import join
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

tree = ET.parse(join("data", "rest_final.xml"))
root = tree.getroot()

reviews = root.findall("Review")
sentences = root.findall("**/sentence")
print("# Reviews   : ", len(reviews))
print("# Sentences : ", len(sentences))
# count = 0
# count_1 = 0
datas = []
categories = []
for i in root.iter('sentence'):
    # count_1+=1
    # print("la: " + str(count_1))
    if(i.get('OutOfScope') != 'TRUE'):
        # count+=1
        opinion = i.find('Opinion')
        # print(i.get('id'))
        # print(opinion.attrib['category'])
        # print(count)
        # if(opinion.attrib['category'] == 'REST#AMBIENCE' or opinion.attrib['category'] == 'REST#PRICES'):
        text = i.find('text')
        datas.append(text.text)
        categories.append(opinion.attrib['category'])
# print(datas)
# print(categories)
df = pd.DataFrame({"datas": datas, "categories": categories})
print(datas)
print(df)
X_train, X_valid, y_train, y_valid = train_test_split(datas, categories, test_size=0.2, random_state=50)
print(y_valid)
vectorizer = CountVectorizer()
transformed_x_train = vectorizer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(transformed_x_train)
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])

text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_valid)
print(np.mean(predicted == y_valid))
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)),])

text_clf_svm = text_clf_svm.fit(X_train,y_train)
predicted_SVM = text_clf_svm.predict(X_valid)
print(np.mean(predicted_SVM == y_valid))