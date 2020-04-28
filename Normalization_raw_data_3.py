from one_hot import convert_to_one_hot
class NoramlPic():
   def norm(self,myobj):
        self.X_train, self.Y_train, self.X_test, self.Y_test, self.classes = myobj
        self.X_train=self.X_train/255.
        self.X_test=self.X_test/255.
        self.Y_train=convert_to_one_hot(self.Y_train,len(self.classes)).T
        self.Y_test = convert_to_one_hot(self.Y_test, len(self.classes)).T
        self.len_class=len(self.classes)
        return self.X_train,self.X_test,self.Y_train,self.Y_test,self.classes,self.len_class

   def __str__(self):
        print("-------------------normal data information--------------------")
        return f'X_train shape {self.X_train.shape}\nY_train shape{self.Y_train.shape}\nX_test shape{self.X_test.shape}\nY_test shape {self.Y_train.shape}\nClasses is: {self.classes}\nnumber of training examples= {self.X_train.shape[0]}\nnumber of test examples= {self.X_test.shape[0]}\nnumber of classes={self.len_class}'
   def __repr__(self):
        return f'{self.X_train}{self.Y_train}{self.X_test}{self.Y_test}{self.classes}{self.X_train.shape[0]}{self.X_test.shape[0]}{self.len_class}'



