import pickle
from os.path import join
from envs.ScrapyEnviroment.Lib import os

import PreProcessing_valid


def Classification():
    # s = "Đồ ăn tại quán ăn rất là đầy đặn,đậm đà,ngon, không gian quán đẹp"
    # s= "Mình thấy suất XL ở đây to hơn, ngon hơn và rất đẹp"
    s="Đường vào quán vẫn khó tìm , lòng_vòng lắm ."
    s = PreProcessing_valid.PreProcessing(s)
    print(s)
    pre = []
    pre.append(s)
    list_file = os.listdir("models_new")
    print(list_file)
    for i in list_file:
        load_file = open(join("models_new",i),'rb')
        clf = pickle.load(load_file)
        t = clf.predict(pre)
        print(t)



    # datas_valid = []
    # labels_valid = []
    # vectorizer = CountVectorizer()
    # transformed_x_valid = vectorizer.fit_transform(s).toarray()
    # load_file = open(join("models","STYLEOPTIONS_new.pkl"),'rb')
    # clf = pickle.load(load_file)
    # print("Loading file : ",clf)
    #
    # with open(join("data_valid", "datas_STYLEOPTIONS_valid_new.txt"), 'r', encoding='utf-8')as file:
    #     for i in file:
    #         datas_valid.append(i)
    # with open(join("data_valid", "labels_STYLEOPTIONS_valid_new.txt"), 'r', encoding='utf-8')as file:
    #     for i in file:
    #         labels_valid.append(i)
    #
    # X_valid = datas_valid
    # a = clf.predict(X_valid)
    # with open("predict.txt",'w',encoding='utf-8') as f:
    #     for i in a:
    #         f.write(i)
    # t = clf.predict(pre)
    # print(t)
    # print(a)
    # print(confusion_matrix(labels_valid, a))

    # X_train, y_train = LoadData("../data_train/.txt", "labels_new1.txt")

if __name__ == "__main__":
    Classification()

