class Compilemodel():
    def compilemodel(self,model_):
        model_.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model_
