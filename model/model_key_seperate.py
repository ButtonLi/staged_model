import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping

# used to load from file
def load_data():
    data=np.loadtxt('5_top_one_million.csv',dtype=np.str,delimiter=" ")
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

max_addr = Y_temp.max()
# # 求得最大地址，并将地址平分成model大小的数量
# block_size=int(max_addr/10)
# Y_list=list(Y_temp)


block_size=int(len(Y_temp)/10)
key_seperator=[]
for i in range(10):
    if i<9:
        init=100000*(i+1)
        key_seperator.append(X_temp[init,0])
    else:
        init=999999
        key_seperator.append(X_temp[init, 0])
print(key_seperator)



X,Y=X_temp,Y_temp

# test data generation
X_test1,Y_test1=X[1000:2000],Y[1000:2000]
X_test2,Y_test2=X[101000:102000],Y[101000:102000]
X_test3,Y_test3=X[501000:502000],Y[501000:502000]
X_test4,Y_test4=X[801000:802000],Y[801000:802000]
X_test5,Y_test5=X[901000:902000],Y[901000:902000]

X_test=np.vstack((X_test1,X_test2,X_test3,X_test4,X_test5))
Y_test=np.hstack((Y_test1,Y_test2,Y_test3,Y_test4,Y_test5))

print("X_test:",X_test)
print("Y_test:",Y_test)



# build a neural network from the 1st layer to the last layer
model1 = Sequential()
model1.add(Dense(units=32, input_dim=5, activation='relu'))
# model1.add(Dense(units=16,activation='relu'))
model1.add(Dense(1))
# choose loss function and optimizing method
adam=optimizers.Adam(lr=0.01)
model1.compile(loss='mse', optimizer=adam)

# Second stage---Including three models
model2_1=Sequential()
model2_1.add(Dense(input_dim=5,units=32,activation='relu'))
model2_1.add(Dense(1))
model2_1.compile(loss='mse',optimizer='adam')

model2_2=Sequential()
model2_2.add(Dense(input_dim=5,units=32,activation='relu'))
model2_2.add(Dense(1))
model2_2.compile(loss='mse',optimizer='adam')

model2_3=Sequential()
model2_3.add(Dense(input_dim=5,units=32,activation='relu'))
model2_3.add(Dense(1))
model2_3.compile(loss='mse',optimizer='adam')

model2_4=Sequential()
model2_4.add(Dense(input_dim=5,units=32,activation='relu'))
model2_4.add(Dense(1))
model2_4.compile(loss='mse',optimizer='adam')

model2_5=Sequential()
model2_5.add(Dense(input_dim=5,units=32,activation='relu'))
model2_5.add(Dense(1))
model2_5.compile(loss='mse',optimizer='adam')

model2_6=Sequential()
model2_6.add(Dense(input_dim=5,units=32,activation='relu'))
model2_6.add(Dense(1))
model2_6.compile(loss='mse',optimizer='adam')

model2_7=Sequential()
model2_7.add(Dense(input_dim=5,units=32,activation='relu'))
model2_7.add(Dense(1))
model2_7.compile(loss='mse',optimizer='adam')

model2_8=Sequential()
model2_8.add(Dense(input_dim=5,units=32,activation='relu'))
model2_8.add(Dense(1))
model2_8.compile(loss='mse',optimizer='adam')

model2_9=Sequential()
model2_9.add(Dense(input_dim=5,units=32,activation='relu'))
model2_9.add(Dense(1))
model2_9.compile(loss='mse',optimizer='adam')

model2_10=Sequential()
model2_10.add(Dense(input_dim=5,units=32,activation='relu'))
model2_10.add(Dense(1))
model2_10.compile(loss='mse',optimizer='adam')
# training
print('Training -----------')
# first stage model training
# model1 training
model1.load_weights('model1_weights.h5')
pre_loss=0

for step in range(100):
    cost=model1.fit(X,Y,batch_size=1000,epochs=20,verbose=2)
    loss=cost.history['loss'][-1]
    if abs(loss-pre_loss)< 10:
        break
    else:
        pre_loss=loss
print("cost:",loss,"\n-------------------------------------------model1 training finished-----------------------------------")
# model1.save_weights('model1_weights.h5')

# second stage model training
# model2_1 training
X_2_1 = X[:block_size]
Y_2_1 = Y[:block_size]

model2_1.load_weights('model1_weights.h5')
print("\n\n model2_1 training...\n\n")
pre_loss=0
for step in range(1000):
    cost=model2_1.fit(X_2_1,Y_2_1,batch_size=100,epochs=20,verbose=2)
    loss=cost.history['loss'][-1]
    if abs(loss-pre_loss)< 10:
        break
    else:
        pre_loss=loss
print("cost:",loss,"\n-------------------------------------------model2_1 training finished-----------------------------------")

# # model2_2 training
X_2_2 = X[block_size:block_size*2]
Y_2_2 = Y[block_size:block_size*2]
model2_2.load_weights('model1_weights.h5')
print("\n\n model2_2 training...\n\n")
pre_loss=0
for step in range(100):
    cost=model2_2.fit(X_2_2, Y_2_2, batch_size=500, epochs=20,verbose=2)
    loss = cost.history['loss'][-1]
    if abs(loss-pre_loss)< 10:
        break
    else:
        pre_loss=loss
print("cost:",loss,"\n-------------------------------------------model2_2 training finished-----------------------------------")

# # model2_3 training
X_2_3 = X[block_size*2:block_size*3]
Y_2_3 = Y[block_size*2:block_size*3]
model2_3.load_weights('model1_weights.h5')
print("\n\n model2_3 training...\n\n")
pre_loss=0
for step in range(100):
    cost=model2_3.fit(X_2_3, Y_2_3, batch_size=500, epochs=20, verbose=2)
    loss = cost.history['loss'][-1]
    if abs(loss-pre_loss)< 10:
        break
    else:
        pre_loss=loss
print("cost:",loss,"\n-------------------------------------------model2_3 training finished-----------------------------------")
# model2_4 training
X_2_4 = X[block_size*3:block_size*4]
Y_2_4 = Y[block_size*3:block_size*4]
model2_4.load_weights('model1_weights.h5')
print("\n\n model2_4 training...\n\n")
pre_loss=0
for step in range(100):
    cost=model2_4.fit(X_2_4, Y_2_4, batch_size=500, epochs=20, verbose=2)
    loss = cost.history['loss'][-1]
    if abs(loss-pre_loss)< 10:
        break
    else:
        pre_loss=loss
print("cost:",loss,"\n-------------------------------------------model2_4 training finished-----------------------------------")

# model2_5 training
X_2_5 = X[block_size*4:block_size*5]
Y_2_5 = Y[block_size*4:block_size*5]
model2_5.load_weights('model1_weights.h5')
print("\n\n model2_5 training...\n\n")
pre_loss=0
for step in range(100):
    cost=model2_5.fit(X_2_5, Y_2_5, batch_size=500, epochs=20, verbose=2)
    loss = cost.history['loss'][-1]
    if abs(loss-pre_loss)< 10:
        break
    else:
        pre_loss=loss
print("cost:",loss,"\n-------------------------------------------model2_5 training finished-----------------------------------")

# model2_6 training
X_2_6 = X[block_size*5:block_size*6]
Y_2_6 = Y[block_size*5:block_size*6]
model2_6.load_weights('model1_weights.h5')
print("\n\n model2_6 training...\n\n")
pre_loss=0
for step in range(100):
    cost=model2_6.fit(X_2_6, Y_2_6, batch_size=500, epochs=20, verbose=2)
    loss = cost.history['loss'][-1]
    if abs(loss-pre_loss)< 10:
        break
    else:
        pre_loss=loss
print("cost:",loss,"\n-------------------------------------------model2_6 training finished-----------------------------------")

# model2_7 training
X_2_7 = X[block_size*5:block_size*6]
Y_2_7 = Y[block_size*5:block_size*6]
model2_7.load_weights('model1_weights.h5')
print("\n\n model2_7 training...\n\n")
pre_loss=0
for step in range(100):
    cost=model2_7.fit(X_2_7, Y_2_7, batch_size=500, epochs=20, verbose=2)
    loss = cost.history['loss'][-1]
    if abs(loss-pre_loss)< 10:
        break
    else:
        pre_loss=loss
print("cost:",loss,"\n-------------------------------------------model2_7 training finished-----------------------------------")


# model2_8 training
X_2_8 = X[block_size*7:block_size*8]
Y_2_8 = Y[block_size*7:block_size*8]
model2_8.load_weights('model1_weights.h5')
print("\n\n model2_8 training...\n\n")
pre_loss=0
for step in range(100):
    cost=model2_8.fit(X_2_8, Y_2_8, batch_size=500, epochs=20, verbose=2)
    loss = cost.history['loss'][-1]
    if abs(loss-pre_loss)< 10:
        break
    else:
        pre_loss=loss
print("cost:",loss,"\n-------------------------------------------model2_8 training finished-----------------------------------")

# model2_9 training
X_2_9 = X[block_size*8:block_size*9]
Y_2_9 = Y[block_size*8:block_size*9]
model2_9.load_weights('model1_weights.h5')
print("\n\n model2_9 training...\n\n")
pre_loss=0
for step in range(100):
    cost=model2_9.fit(X_2_9, Y_2_9, batch_size=500, epochs=20, verbose=2)
    loss = cost.history['loss'][-1]
    if abs(loss-pre_loss)< 10:
        break
    else:
        pre_loss=loss
print("cost:",loss,"\n-------------------------------------------model2_9 training finished-----------------------------------")

# model2_10 training
X_2_10 = X[block_size*9:]
Y_2_10 = Y[block_size*9:]
model2_10.load_weights('model1_weights.h5')
print("\n\n model2_10 training...\n\n")
pre_loss=0
for step in range(100):
    cost=model2_10.fit(X_2_10, Y_2_10, batch_size=500, epochs=20, verbose=2)
    loss = cost.history['loss'][-1]
    if abs(loss-pre_loss)< 10:
        break
    else:
        pre_loss=loss
print("cost:",loss,"\n------------------------------------------- model2_10 training finished-----------------------------------")

# testing
print('\nTesting ------------')

for i in range(len(X_test)):
    list_t=[]
    for j in range(len(X_test[i])):
        list_t.append(X_test[i][j])
    list_t = [list_t]
    list_t = np.array(list_t)
    pred=model1.predict(list_t)
    f=open('5_test_reverse.txt','a+')
    f.write(str(Y_test[i])+" "+str(int(pred[0][0]))+" ")

    if list_t[0][0] < key_seperator[0]:
        pos=model2_1.predict(list_t)
        f.write(str(int(pos[0][0]))+"\n")
    if key_seperator[0] <= list_t[0][0] < key_seperator[1]:
        pos2 = model2_2.predict(list_t)
        f.write(str(int(pos2[0][0])) + "\n")
    if key_seperator[1] <= list_t[0][0] < key_seperator[2]:
        pos = model2_3.predict(list_t)
        # print(pos)
        f.write(str(int(pos[0][0])) + "\n")
    if key_seperator[2] <= list_t[0][0] < key_seperator[3]:

        pos2 = model2_4.predict(list_t)
        f.write(str(int(pos2[0][0])) + "\n")
    if key_seperator[3] <= list_t[0][0] < key_seperator[4]:
        pos=model2_5.predict(list_t)
        # print(pos)
        f.write(str(int(pos[0][0])) + "\n")
    if key_seperator[4] <= list_t[0][0] < key_seperator[5]:
        pos2=model2_6.predict(list_t)
        f.write(str(int(pos2[0][0])) + "\n")

    if key_seperator[5] <= list_t[0][0] < key_seperator[6]:
        pos = model2_7.predict(list_t)
        # print(pos)
        f.write(str(int(pos[0][0])) + "\n")

    if key_seperator[6] <= list_t[0][0] < key_seperator[7]:
        pos2=model2_8.predict(list_t)
        f.write(str(int(pos2[0][0])) + "\n")

    if key_seperator[7] <= list_t[0][0] < key_seperator[8]:
        pos = model2_9.predict(list_t)
        # print(pos)
        f.write(str(int(pos[0][0])) + "\n")

    if key_seperator[8] <= list_t[0][0] < key_seperator[9]:
        pos2 = model2_10.predict(list_t)
        f.write(str(int(pos2[0][0])) + "\n")



