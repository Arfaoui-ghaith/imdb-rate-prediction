from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class KNeighbors:

    def __init__(self, dataset):
        
        x=dataset.drop('Rating',axis=1).drop('Linear_Rating',axis=1)
        y=dataset[['Rating']]
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
      

    def predict(self,list):
        neigh = KNeighborsClassifier(n_neighbors=21)
        neigh.fit(self.x_train,self.y_train)
        score = neigh.score(self.x_test,self.y_test)
        res = neigh.predict([list])[0]
        return score,res

