#!/usr/bin/env python
# coding: utf-8

# ## 达观杯2021

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append("./")


# ### 加载数据集，并切分train/dev

# In[2]:


# 加载数据
df_train = pd.read_csv("./datasets/phase_1/splits/fold_0/train.txt")
df_train.columns = ["id", "text", "label"]
df_val = pd.read_csv("./datasets/phase_1/splits/fold_0/dev.txt")
df_val.columns = ["id", "text", "label"]
df_test = pd.read_csv("./datasets/phase_1/splits/fold_0/test.txt")
df_test.columns = ["id", "text", ]

# 构建词表
charset = set()
for text in df_train['text']:
    for char in text.split(" "):
        charset.add(char)
id2char = ['OOV', '，', '。', '！', '？'] + list(charset)
char2id = {id2char[i]: i for i in range(len(id2char))}

# 标签集
id2label = list(df_train['label'].unique())
label2id = {id2label[i]: i for i in range(len(id2label))}


# ### 定义模型

# In[3]:


# 定义模型

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
MAX_LEN = 128
input_layer = Input(shape=(MAX_LEN,))
layer = Embedding(input_dim=len(id2char), output_dim=256)(input_layer)
layer = Bidirectional(LSTM(256, return_sequences=True))(layer)
layer = Flatten()(layer)
output_layer = Dense(len(id2label), activation='softmax')(layer)
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ### 准备输入数据
# 
# 对训练集、验证集、测试集进行输入转换，构造模型输入。

# In[4]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

X_train, X_val, X_test = [], [], []
y_train = np.zeros((len(df_train), len(id2label)), dtype=np.int8)
y_val = np.zeros((len(df_val), len(id2label)), dtype=np.int8)

for i in range(len(df_train)):
    X_train.append([char2id[char] for char in df_train.loc[i, 'text'].split(" ")])
    y_train[i][label2id[df_train.loc[i, 'label']]] = 1
for i in range(len(df_val)):
    X_val.append([char2id[char] if char in char2id else 0 for char in df_val.loc[i, 'text'].split(" ")])
    y_val[i][label2id[df_val.loc[i, 'label']]] = 1
for i in range(len(df_test)):
    X_test.append([char2id[char] if char in char2id else 0 for char in df_test.loc[i, 'text'].split(" ")])

X_train = pad_sequences(X_train, maxlen=MAX_LEN, padding='post', truncating='post')
X_val = pad_sequences(X_val, maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=MAX_LEN, padding='post', truncating='post')


# ### 模型训练

# In[5]:


model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)


# In[19]:


y_val_pred = model.predict(X_val).argmax(axis=-1)
print(y_val_pred[: 20])
y_val = []
for i in range(len(df_val)):
    y_val.append(label2id[df_val.loc[i, 'label']])
y_val = [int(w) for w in y_val]
print(y_val[: 20])

from sklearn.metrics import classification_report
results = {}
classification_report_dict = classification_report(y_val_pred, y_val, output_dict=True)
for key0, val0 in classification_report_dict.items():
    if isinstance(val0, dict):
        for key1, val1 in val0.items():
            results[key0 + "__" + key1] = val1

    else:
        results[key0] = val0

import json
print(json.dumps(results, indent=2, ensure_ascii=False))


# ### 输出预测结果

# In[7]:


y_pred = model.predict(X_test).argmax(axis=-1)
pred_labels = [id2label[i] for i in y_pred]
pd.DataFrame({"id": df_test['id'], "label": pred_labels}).to_csv("submission.csv", index=False)



# 提交结果：
# 0.36730954652


# In[ ]:




