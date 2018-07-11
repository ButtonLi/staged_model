import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pickle

def load_data():
    data = np.loadtxt('../data/50000.csv', dtype=np.str, delimiter="\t")
    return data

def store_tree(tree, filename):
    fw = open(filename, 'wb')
    fw.write(pickle.dumps(tree))
    fw.close()

def grab_tree(filename):
    fr = open(filename,'rb')
    data = pickle.loads(fr.read())
    fr.close()
    return data

if __name__ == "__main__":
    data=load_data()
    lists=[]
    for i in range(len(data)):
        items=data[i].split(" ")
        lists.append(int(items[0]))
        lists.append(int(items[1]))

    lists=np.array(lists)
    lists=lists.reshape(-1,2)
    X_temp = lists[:, -1]
    Y_temp=lists[:,0]
    print(Y_temp)
    # X_temp_min, X_temp_max = X_temp.min(), X_temp.max()
    # X_temp = (X_temp - X_temp_min) / (X_temp_max - X_temp_min)
    x=X_temp.reshape(-1,1)
    y=Y_temp
    print(x)

    # reg = DecisionTreeRegressor(criterion='mse', max_depth=17)
    # dt = reg.fit(x, y)
    # store_tree(dt, '../tree_folder/dtree.txt')
    dt=grab_tree("../tree_folder/dtree.txt")
    x_test = X_temp[:5000]
    y_test= Y_temp[:5000]
    for i in range(len(x_test)):
        test_item=np.array([x_test[i]])
        test_item=test_item.reshape(-1,1)
        y_pred = dt.predict(test_item)
        print(y_test[i],"  ")
        print(y_pred)
