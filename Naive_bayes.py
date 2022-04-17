from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

class Naive_bayes:

    def __init__(self, dataset):
        
        x=dataset.drop('Rating',axis=1).drop('Linear_Rating',axis=1)
        y=dataset[['Rating']]
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
      

    def predict(self,list):
        gnb = GaussianNB()
        gnb.fit(self.x_train,self.y_train)
        score = gnb.score(self.x_test,self.y_test)
        res = gnb.predict([list])[0]
        return score,res

