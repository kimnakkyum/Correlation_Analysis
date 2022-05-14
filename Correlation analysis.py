#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
iris = load_iris()
iris


# In[2]:


# dict 형태의 데이터를 보기 좋게 만들기 위해 데이터 프레임 타입으로 변환합니다.
import pandas as pd

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['class'] = iris.target

df.tail()


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


np.unique(df["class"], return_counts = True)


# In[14]:


# 0으로 labeling된 꽃과 1로 labeling된 꽃의 상관성 분석


# In[5]:


df_0 = df[df["class"] == 0] 


# In[6]:


df_1 = df[df["class"] == 1] 


# In[7]:


df_01 = pd.concat([df_0, df_1])


# In[15]:


# pearson 상관계수


# In[8]:


df_corr = df_01.corr()
df_corr


# In[11]:


# 종속 변수인 class와 나머지 독립 변수들과의 상관계수를 확인하기


# In[10]:


df_corr.iloc[:-1, -1]


# In[17]:


# sepal length와 꽃의 종류와 양의 상관관계를 나타냄
# sepal width와 꽃의 종류와 음의 상관관계를 나타냄
# petal lenghth와 꽃의 종류와 강한 양의 상관관계를 나타냄
# petal width와 꽃의 종류와 강한 양의 상관관계를 나타냄


# In[18]:


# pearson 상관계수를 이용하여 상관성을 분석함.
# -1에 가까울수록 음의 상관관계를 나타내고, 1에 가까울수록 양의 상관관계를 나타낸다.
# 0에 가까울수록 상관성이 없다.


# In[ ]:




