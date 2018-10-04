import pandas as pd
from os.path import join
from lxml import etree as ET
from pyvi import ViTokenizer
from Sem_eval_Bao import StopWord1

tree = ET.parse(join("data", "rest_final_new.xml"))
root = tree.getroot()
datas = []
categories = []
reviews = root.findall("Review")
sentences = root.findall("**/sentence")
print("# Reviews   : ", len(reviews))
print("# Sentences : ", len(sentences))

count = 0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0

#Counter({'REST#QUALITY': 1032, 'REST#SERVICE': 353, 'REST#GENERAL': 337, 'REST#PRICES': 322, 'REST#AMBIENCE': 305, 'REST#STYLEOPTIONS': 292, 'REST#LOCATION': 146})
list = []
for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        opinion = i.find('Opinion')
        name = opinion.attrib['category']
        # print(name)
        list.append(name)
        if (name == 'REST#SERVICE'):
            text = i.find('text')
            datas.append(text.text)
            categories.append(opinion.attrib['category'])
            count += 1

print(count)
for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        opinion = i.find('Opinion')
        if (opinion.attrib['category'] == 'REST#QUALITY' and count1<100):
            text = i.find('text')
            text = text.text
            datas.append(text)
            categories.append('None')
            count1 +=1

for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        opinion = i.find('Opinion')
        if (opinion.attrib['category'] == 'REST#GENERAL' and count2<100):
            text = i.find('text')
            text = text.text
            datas.append(text)
            categories.append('None')
            count2+=1

for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        opinion = i.find('Opinion')
        if (opinion.attrib['category'] == 'REST#PRICES' and count3<100):
            text = i.find('text')
            text = text.text
            datas.append(text)
            categories.append('None')
            count3 +=1

for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        opinion = i.find('Opinion')
        if (opinion.attrib['category'] == 'REST#AMBIENCE' and count4<100):
            text = i.find('text')
            text = text.text
            datas.append(text)
            categories.append('None')
            count4+=1

for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        opinion = i.find('Opinion')
        if (opinion.attrib['category'] == 'REST#STYLEOPTIONS' and count5<100):
            text = i.find('text')
            text = text.text
            datas.append(text)
            categories.append('None')
            count5+=1
print(count3)

for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        opinion = i.find('Opinion')
        if (opinion.attrib['category'] == 'REST#LOCATION' and count6<100):
            text = i.find('text')
            text = text.text
            datas.append(text)
            categories.append('None')
            count6+=1

SPECIAL_CHARACTER = '%@$=+-!;🏻/()👍*❤"😍&^:♥<>#|\n\t\''
with open(join("data_train", "datas_PRICES_new.txt"),'w',encoding='utf-8') as file:
    for i in datas:
        my_words = i.split(" ")
        for word1 in i:
            if word1 in SPECIAL_CHARACTER:
                i = i.replace(word1, "")
                i = i.replace("  ", " ")
        for word in my_words:
            if len(word) > 20 or len(word) < 2:
                i = i.replace(word, "")
                i = i.replace("  ", " ")
        i = ViTokenizer.tokenize(i)
        my_words = i.split(" ")
        for word in my_words:
            if word in StopWord1.STOP_WORDS:
                i = i.replace(word, "")
                i = i.replace("  ", " ")
        i = i.lower()
        file.write(i+"\n")

with open(join("data_train", "labels_PRICES_new.txt"),'w',encoding='utf-8') as file:
    for i in categories:
        file.write(str(i)+"\n")

print(len(datas))
print(len(categories))
