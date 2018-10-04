from pyvi import ViTokenizer

datas = []
with open("data_bao_final.txt",'r',encoding='utf-8') as f:
    for i in f:
        datas.append(i)

SPECIAL_CHARACTER = '%@$=+-!;🏻/()👍*❤"😍&^:♥<>#|\n\t\''
with open("data_bao_final_new_seg.txt",'w',encoding='utf-8') as file:
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
        # my_words = i.split(" ")
        # for word in my_words:
        #     if word in StopWords.STOP_WORDS:
        #         i = i.replace(word, "")
        #         i = i.replace("  ", " ")
        i = i.lower()
        file.write(i+"\n")