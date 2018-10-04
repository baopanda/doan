
import pandas as pd
from os.path import join
from lxml import etree as ET
from pyvi import ViTokenizer
import StopWords

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
for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        for opi in i.iter('Opinion'):
            name = opi.get('category')
            if (name == 'REST#LOCATION'):
                text = i.find('text').text
                datas.append(text)
                categories.append(name)
                count += 1
print(count)
for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        for opi in i.iter('Opinion'):
            name = opi.get('category')
            if (name == 'REST#QUALITY' and count1<32):
                text = i.find('text').text
                if(text not in datas):
                    datas.append(text)
                    categories.append('None')
                    count1 += 1

for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        for opi in i.iter('Opinion'):
            name = opi.get('category')
            if (name == 'REST#SERVICE' and count2<32):
                text = i.find('text').text
                if (text not in datas):
                    datas.append(text)
                    categories.append('None')
                    count2 += 1

for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        for opi in i.iter('Opinion'):
            name = opi.get('category')
            if (name == 'REST#GENERAL' and count3<32):
                text = i.find('text').text
                if (text not in datas):
                    datas.append(text)
                    categories.append('None')
                    count3 += 1

for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        for opi in i.iter('Opinion'):
            name = opi.get('category')
            if (name == 'REST#PRICES' and count4<32):
                text = i.find('text').text
                if (text not in datas):
                    datas.append(text)
                    categories.append('None')
                    count4 += 1

for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        for opi in i.iter('Opinion'):
            name = opi.get('category')
            if (name == 'REST#AMBIENCE' and count5<32):
                text = i.find('text').text
                if (text not in datas):
                    datas.append(text)
                    categories.append('None')
                    count5 += 1

for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        for opi in i.iter('Opinion'):
            name = opi.get('category')
            if (name == 'REST#STYLEOPTIONS' and count6<32):
                text = i.find('text').text
                if (text not in datas):
                    datas.append(text)
                    categories.append('None')
                    count6 += 1

SPECIAL_CHARACTER = '%@$=+-!;🏻/()👍*❤"😍&^:♥<>#|\n\t\''

with open(join("data_train", "datas_LOCATION_new.txt"),'w',encoding='utf-8') as file:
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
            if word in StopWords.STOP_WORDS:
                i = i.replace(word, "")
                i = i.replace("  ", " ")
        i = i.lower()
        file.write(i+"\n")

with open(join("data_train", "labels_LOCATION_new.txt"),'w',encoding='utf-8') as file:
    for i in categories:
        file.write(str(i)+"\n")

print(len(datas))
print(len(categories))


