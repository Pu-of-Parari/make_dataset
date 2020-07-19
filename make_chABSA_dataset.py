import pandas as pd
from pandas import DataFrame
import numpy as np
import json 
import glob
import collections
import MeCab
import csv
from sklearn.model_selection import train_test_split
import pprint

"""reshape chABSA-dataset for classification script
"""

def wakati(input_str):
    '''分かち書き用関数
    args << input_str : 入力テキスト
    return >> m.parse(wakatext) : 分かち済みテキスト'''
    wakatext = input_str
    m = MeCab.Tagger('-Owakati')
    #print(m.parse(wakatext))
    tokenized = m.parse(wakatext).replace("\n", "")

    return tokenized

def space_tokenize(input_str):
    tokenized = ""
    for char in input_str:    
        tokenized += char + " "

    return tokenized[:-1]


def extract_category(file_list):
    cat17_list = list()
    cat33_list = list()
    data_cat_list = list()

    for file in file_list:
        if file.find(".json") == -1:
            continue

        else:
            with open(file, "br") as f:
                j = json.load(f)
            
            header = j["header"]
            sentences = j["sentences"]

            category17 = header["category17"]
            category33 = header["category33"]
            cat17_list.append(category17)
            cat33_list.append(category33)

            for obj in sentences:
                s = obj["sentence"].replace("\u3000", "").replace("\n","").replace("　","").replace("\t","")
                data_cat_list.append([s,category17,category33])
    
    cat17_dic = make_label_dic(set(cat17_list))
    cat33_dic = make_label_dic(set(cat33_list))

    return data_cat_list, cat17_dic, cat33_dic


def create_rating(sentences):
    rating = []
    for obj in sentences:
        s = obj["sentence"].replace("\u3000", "").replace("\n","").replace("　","").replace("\t","")
        op  = obj["opinions"]
        polarity = 0
        for o in op:
            p = o["polarity"]
            if p == "positive":
                polarity += 1
            elif p == "negative":
                polarity -= 1
        
        rating.append([s,polarity])
    return rating

def sent_scoring(file_list):
    rating = list()
    for file in file_list:
        if file.find(".json") == -1:
            continue
        else:
            with open(file, "br") as f:
                j = json.load(f)
            sentences = j["sentences"]

            rating += create_rating(sentences)

    return rating

def create_negpos(data):
    neg = list()
    pos = list()
    neutral = list()
    all_nnp = list()

    for d in data:
        if d[1] > 0:
            pos.append([d[0], "positive"])
            all_nnp.append([d[0], "positive"])
        elif d[1] == 0:
            neutral.append([d[0], "neutral"])
            all_nnp.append([d[0], "neutral"])
        else:
            neg.append([d[0], "negative"])
            all_nnp.append([d[0], "negative"])
    
    print("positive:", len(pos))
    print("neutral:", len(neutral))
    print("negative:", len(neg))

    return all_nnp







def make_label_dic(cat_list):
    label_dic = dict()
    for i, cat in enumerate(cat_list):
        label_dic[cat] = i
    
    return label_dic


def split_data(file_name):
    '''データをtrain, dev, testに分割する
    '''
    dataset = pd.read_table(file_name, header=None)
    sent = dataset[0]
    label = dataset[1]
    #print(label)

    sent_train, sent_devtest, label_train, label_devtest = train_test_split(sent, label, test_size=0.2, random_state=69, stratify=label)
    sent_dev, sent_test, label_dev, label_test = train_test_split(sent_devtest, label_devtest, test_size=0.5, random_state=69, stratify=label_devtest)
    
    ds_train = pd.concat([sent_train, label_train], axis=1)
    ds_dev = pd.concat([sent_dev, label_dev], axis=1)
    ds_test = pd.concat([sent_test, label_test], axis=1)

    #print(label_train.value_counts() / len(label_train))
    #print(label_dev.value_counts() / len(label_dev))
    #print(label_test.value_counts() / len(label_test))
    
    return ds_train, ds_dev, ds_test


def write_split_file(data_dir, train_data, dev_data, test_data):
    train_data_dir = data_dir.replace(".txt", "_train.txt")
    dev_data_dir = data_dir.replace(".txt", "_dev.txt")
    test_data_dir = data_dir.replace(".txt", "_test.txt")

    train_data.to_csv(train_data_dir, sep="\t", header=False, index=False)
    dev_data.to_csv(dev_data_dir, sep="\t", header=False, index=False)
    test_data.to_csv(test_data_dir, sep="\t", header=False, index=False)



def write_all_file(data_dir, data, label_dic):
    tok_data_dir = data_dir.replace(".txt", "_tok.txt")

    if data_dir.find("33") != -1: cat33_flag = True
    else: cat33_flag = False

    with open(data_dir, "w") as outf, open(tok_data_dir, "w") as outf_t:
        writer = csv.writer(outf, delimiter='\t', lineterminator='\n')
        writer_t = csv.writer(outf_t, delimiter='\t', lineterminator='\n')
        #writer.writerow(["sent", "label"])

        i = 0 
        data_list = []
        for data_ in data:
            i += 1
            sent = "<s> " + data_[0] + " </s>"
            sent_t = "<s> " + wakati(data_[0]) + " </s>"
            sent_s = "<s> " + space_tokenize(data_[0]) + " </s>"

            data_list.append(label_dic.get(data_[1]))
            if cat33_flag: label_id = str(label_dic.get(data_[2])) # flagがTrueならカテゴリ33(data_[2])をラベルにする
            else: label_id = str(label_dic.get(data_[1]))

            data = [sent, label_id]
            data_t = [sent_t, label_id]
            data_s = [sent_s, label_id]


            writer.writerow(data)
            writer_t.writerow(data_s)

    print(data_dir)
    print(i,"samples have written:)")

    return tok_data_dir

def make_dataset(data_dir, data, label_dic):
    #DATASET_CAT17_FILE = "./cat17.txt"
    tok_data_dir = write_all_file(data_dir, data, label_dic)

    train_data, dev_data, test_data = split_data(data_dir)
    print("train:", len(train_data))
    print("dev:", len(dev_data))
    print("test:", len(test_data))
    write_split_file(data_dir, train_data, dev_data, test_data)
    
    train_t_data, dev_t_data, test_t_data = split_data(tok_data_dir)
    write_split_file(tok_data_dir, train_t_data, dev_t_data, test_t_data)




if __name__ == '__main__':
    data_dir = "chABSA-dataset_raw/chABSA-dataset/*.json"
    file_list = glob.glob(data_dir)

    
    # categoryファイル書き出し
    cat_all_data, cat17_dic, cat33_dic = extract_category(file_list)

    DATASET_CAT17_FILE = "./chABSA-dataset/cat17.txt"
    make_dataset(DATASET_CAT17_FILE, cat_all_data, cat17_dic)

    DATASET_CAT33_FILE = "./chABSA-dataset/cat33.txt"
    make_dataset(DATASET_CAT33_FILE, cat_all_data, cat33_dic)

    # nnpファイル書き出し
    nnp_dic = {"positive":2, "neutral":1, "negative":0}
    
    data_rating = sent_scoring(file_list)
    all_nnp_data = create_negpos(data_rating)
    
    DATASET_NNP_FILE = "./chABSA-dataset/nnp.txt"
    make_dataset(DATASET_NNP_FILE, all_nnp_data, nnp_dic)
    
