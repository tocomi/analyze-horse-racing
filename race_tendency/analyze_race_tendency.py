# -*- coding: utf-8 -*-

'''
重賞レース傾向の解析スクリプト

過去の重賞の結果から傾向を分析する
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation
import sys

def usage():
    usage = """
    !!! Argument is Illegal !!!
    
    - Usage
    python analyze_race_tendency [csv_file_name]
    """
    print(usage)

def parse_args():
    # 引数が不足していたら実行例を出して終了 #
    if len(sys.argv) <= 1:
        usage()
        exit(9)

    args = {}
    args['csv_file_name'] = sys.argv[1]
    return args

def read_csv(args):
    # CSVファイルの読み込み #
    data = pd.read_csv(args['csv_file_name'])
    # 馬名は解析に不要なので消す #
    del data['horse_name']
    return data
    # return data[data['rank'] <= 3]

def show_graph(data):
    # グラフサイズ設定 #
    plt.figure(figsize=(10,6))
    # X軸,Y軸の設定 #
    plt.hist(data['age'])
    # グラフの表示 #
    plt.show()

def analyze_svm(data):

    data_count = len(data)

    x_data = data.loc[:, ['age', 'female', 'handi', 'weight']].values.reshape(-1, 1)
    y_data = data['rank'].values.reshape(-1, 1)

    kf = cross_validation.KFold(data_count, n_folds=4)

    for train, test in kf:

        x_train = x_data[train]
        x_test = x_data[test]
        y_train = y_data[train]
        y_test = y_data[test]

        model = svm.SVR()
        model.fit(x_train, y_train)
        print("Linear: Training Score = %f, Testing(Validate) Score = %f" % (model.score(x_train, y_train), model.score(x_test, y_test)))

if __name__ == '__main__':
    args = parse_args()
    data = read_csv(args)
    # show_graph(data)
    analyze_svm(data)