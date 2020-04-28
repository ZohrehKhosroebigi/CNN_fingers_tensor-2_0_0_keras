class Trainmodel():

    def trainmodel(self, x_train, y_train,model_, epoch, batch_size):
        model_.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size)
        return model_

