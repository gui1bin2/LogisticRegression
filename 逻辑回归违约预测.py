# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# 导入并处理数据

bankloan = pd.read_csv("gui/bankloan.csv")

# 将教育变量转换成二分类变量，并删除原有多分类变量
bankloan = pd.concat([bankloan,pd.get_dummies(bankloan.教育,drop_first=False)],
               axis=1).drop(['教育'],axis=1)
# 提取建模用数据
model_data = bankloan[:700]
# 提取需要进行预测的数据
predict_data = bankloan[700:]

#变量替换为0,1表示
model_data['违约'] = model_data['违约'].replace(('否','是'),('0','1'))


# 将自变量与因变量分开
x,y = model_data.drop(['违约','ID'],axis=1),model_data[['违约']]
# 随机抽取训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 10)
# 开始构建一个逻辑回归模型
model = LogisticRegression()
# 模型以X_train,y_train为输入数据进行训练
model.fit(x_train,y_train)
# 打印针对测试集而言的准确率
print(accuracy_score(y_test,model.predict(x_test)))
# 使用训练得到模型对这些新申请贷款的人的违约风险进行预测
model.predict(predict_data.drop(['ID','违约'],axis=1))


df = pd.DataFrame(model.predict(predict_data.drop(['ID','违约'],axis=1)))
df.columns = ['违约']

####很重要，重置索引！！！！！不然无法合并
predict_data1 = predict_data.reset_index(drop=True)

#清洗预测集
predict_data1['违约'] = df['违约']
predict_data1 = predict_data1[['ID','违约']]
predict_data1['违约'] = predict_data1['违约'].replace(('0','1'),('否','是'))



#下面的与主题无关，查数据类型用的
print (df.dtypes)
print (model_data.dtypes)
