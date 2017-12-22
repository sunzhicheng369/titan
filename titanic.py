#coding:utf8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

titan = pd.read_csv("./train.csv")


# 对数据进行类别的处理
titan["Age"]=titan["Age"].fillna(titan["Age"].mean())
# titan["Pclass"]=titan["Pclass"].astype('category')
titan["Sex"]=titan["Sex"].astype('category')
titan["Embarked"]=titan["Embarked"].astype('category')
titan["SibSp"]=titan["SibSp"].astype('category')
titan["Parch"]=titan["Parch"].astype('category')


# 画图操作
fig = plt.figure()

plt.subplot(311)
# 取出生存者，和死亡者的年龄，年龄为连续值
age_yes = titan["Age"][titan["Survived"]==1]
age_no = titan["Age"][titan["Survived"]==0]
age_all = [pd.concat([age_yes,age_no]),age_yes]
plt.hist(age_all,5,histtype="bar",label=["all","yes"])
plt.title("the age")
plt.legend()

plt.subplot(312)
plt.title("the Pclass")
age_yes = titan["Pclass"][titan["Survived"]==1]
age_no = titan["Pclass"][titan["Survived"]==0]
age_all = [pd.concat([age_no,age_yes]),age_yes]
plt.hist(age_all,histtype="bar",label=["all","yes"])
plt.legend()


plt.subplot(313)
plt.title("the Sex")
sex_yes = titan["Sex"][titan["Survived"]==1]
sex_no = titan["Sex"][titan["Survived"]==0]
sex_all = [pd.concat([age_no,age_yes]),age_yes]
plt.hist(age_all,histtype="bar",label=["all","yes"])
plt.legend()



plt.show()

emb_s = titan[["Embarked","Survived"]]
emb_s['count']=1
emb_s=pd.pivot_table(emb_s,index="Embarked",columns='Survived',values='count',aggfunc=np.sum)
emb_s[1]=emb_s[1].fillna(0)
emb_s["pres"] = emb_s[1]/(emb_s[1]+emb_s[0])
emb_s[[0,1]].plot(kind="bar")
from sklearn.preprocessing import Imputer