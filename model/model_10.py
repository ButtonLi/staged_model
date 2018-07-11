import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping
import random

# used to load from file
def load_data():
    data=np.loadtxt('../data/1M_snd.csv',dtype=np.str,delimiter=",")
    return data

data=load_data()
lists=[]
for i in range(len(data)):
    lists.append(data[i])
# print(lists)
lists=np.array(lists)
# print(lists)
X_temp=lists[:,0:5].astype(float)

# 数据归一化操作
X_temp[:,0]=(X_temp[:,0]-X_temp[:,0].min())/(X_temp[:,0].max()-X_temp[:,0].min())
X_temp[:,1]=(X_temp[:,1]-X_temp[:,1].min())/(X_temp[:,1].max()-X_temp[:,1].min())
X_temp[:,2]=(X_temp[:,2]-X_temp[:,2].min())/(X_temp[:,2].max()-X_temp[:,2].min())
X_temp[:,3]=(X_temp[:,3]-X_temp[:,3].min())/(X_temp[:,3].max()-X_temp[:,3].min())
X_temp[:,4]=(X_temp[:,4]-X_temp[:,4].min())/(X_temp[:,4].max()-X_temp[:,4].min())


Y_temp=lists[:,5].astype(float)

# 定义第二层模型数量
model_num=10
max_addr = Y_temp.max()
# 求得最大地址，并将地址平分成model大小的数量
block_size=int(max_addr/model_num)


X,Y=X_temp,Y_temp

# test data generation
L=[]
for i in range(1,1000001):
    L.append(i)
slice=random.sample(L,10000)
X_test=[]
Y_test=[]
X_list=X.tolist()
Y_list=Y.tolist()
for item in slice:
    X_test.append(X_list[item-1])
    Y_test.append(Y_list[item - 1])
X_test=np.array(X_test)
Y_test=np.array(Y_test)
print("X_test:",X_test)
print("Y_test:",Y_test)

# build a neural network from the 1st layer to the last layer
# choose loss function and optimizing method
adam=optimizers.Adam(lr=0.01)
model1 = Sequential()
model1.add(Dense(units=32, input_dim=5, activation='relu'))
# model1.add(Dense(units=16,activation='relu'))
model1.add(Dense(1))
model1.compile(loss='mse', optimizer=adam)

# Second stage---Including three models
adam=optimizers.Adam(lr=0.01)
model2_1=Sequential()
model2_1.add(Dense(input_dim=5,units=32,activation='relu'))
model2_1.add(Dense(1))
model2_1.compile(loss='mse',optimizer=adam)

# training
print('Training -----------')
# first stage model training
# model1 training
# model1.load_weights('../weights/model1_weights.h5')
pre_loss=0
for step in range(100):
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    cost=model1.fit(X,Y,batch_size=10000,epochs=20,verbose=2,callbacks=[early_stopping])
    loss=cost.history['loss'][-1]
    if abs(loss-pre_loss)< 10:
        break
    else:
        pre_loss=loss
print("cost:",loss,"\n-------------------------------------------model1 training finished-----------------------------------")
# model1.save_weights('../weights/model1_weights.h5')

for i in range(model_num):
    print("model2_", str(i + 1), "is training")
    train_X = X[i * block_size: (i + 1) * block_size - 1]
    train_Y = Y[i * block_size: (i + 1) * block_size - 1]
    pre_loss = 0
    for step in range(1000):
        cost = model2_1.fit(train_X, train_Y, batch_size=100, epochs=20, verbose=2)
        loss = cost.history['loss'][-1]
        if abs(loss - pre_loss) < 10:
            break
        else:
            pre_loss = loss
    weights = "../weights/model2_"+str(i+1)+"_weights.h5"
    model2_1.save_weights(weights)

for i in range(len(X_test)):
    list_t=[]
    for j in range(len(X_test[i])):
        list_t.append(X_test[i][j])
    list_t = [list_t]
    list_t = np.array(list_t)
    # print(list_t)
    pred = model1.predict(list_t)
    f=open('../results/snd_10model_test2.csv','a+')
    f.write(str(int(Y_test[i]))+" "+str(int(pred[0][0]))+" ")

    # 根据上一层预测，选择下一层模型
    model_id=int( model_num * pred[0][0] / max_addr )+1
    weights="../weights/model2_"+str(model_id)+"_weights.h5"
    print(model_id," model is selected")

    model2_1.load_weights(weights)
    pos = model2_1.predict(list_t)
    f.write(str(int(pos[0][0])) + " " + str(model_id) + "\n")
