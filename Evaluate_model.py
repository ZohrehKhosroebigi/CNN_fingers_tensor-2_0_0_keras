from writing import *
class Evaluate_model():
        def evaluatemodel(self, x_test, y_test,model_):
            preds = model_.evaluate(x=x_test, y=y_test)
            mywriting = Writelogs()
            mywriting.writing("Loss = " + str(preds[0]), "Test Accuracy = " + str(preds[1]))
            # print("------results-------")
            # print("Loss = " + str(preds[0]))
            # print("Test Accuracy = " + str(preds[1]))
            return model_