"""Small sklearn multilayer perceptron forecasts, with explicit history variants."""
from sklearn.neural_network import MLPClassifier
from ._nonlinear import NonlinearModel


class NNModel(NonlinearModel):
    model_name = "nn"

    def estimator(self):
        return MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1.0, max_iter=500,
                             early_stopping=True, random_state=0)
