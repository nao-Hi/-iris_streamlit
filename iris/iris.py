# 必要なライブラリをインポート
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os


# 現在のファイル(iris.py)のあるディレクトリパスを取得
base_dir = os.path.dirname(os.path.abspath(__file__))
# 結合して正しいパスを作る
model_path = os.path.join(base_dir, 'models', 'model_iris.pkl')

with open(model_path, 'rb') as f:
    clf = pickle.load(f)


# 1. 保存されたモデルを読み込む
# 'rb'は読み込みモードを意味します（binary read mode）
# with open('models/model_iris.pkl', 'rb') as f:
    # clf = pickle.load(f)


# 2. Streamlitアプリの設定
# サイドバーにスライダーを作成して花の特徴を入力
# 特徴の中でも分類に影響する sepal_length と petal_length を入力
# 分類にほとんど影響しない sepal_width と petal_width に関しては常に0.0となるように設定
st.sidebar.header('Iris Flower Feature Input')
# 花の特徴をスライダーで入力
sepal_length = st.sidebar.slider('Sepal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = 0.0
petal_width = 0.0
petal_length = st.sidebar.slider('Petal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)


# 3. メインパネルに入力された値を表示
st.title('Iris Classifier')
st.write('## Input Value')


# 4. 入力値をDataFrameに変換
# 入力されたデータを使ってモデルに渡すための形式に変換
value_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                        columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])


# 5. 入力された値を表示
st.write(value_df)


# 6. モデルを使って予測を行う
# predict_probaはそれぞれの花の種類の確率を返す
pred_probs = clf.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs, columns=['setosa', 'versicolor', 'virginica'], index=['probability'])


# 7. 予測結果を表示
st.write('## Prediction')
st.write(pred_df)


# 8. 予測結果を使って最も可能性の高い花の種類を表示
name = pred_df.idxmax(axis=1).tolist()  # 確率が最も高い花の種類を取得
st.write('## Result')
st.write('このアイリスはきっと', str(name[0]), 'です!')