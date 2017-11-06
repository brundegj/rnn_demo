from hyperopt import fmin, tpe
import os


def find_best_model(model_holder, algo=tpe.suggest, max_evals=10):
    return fmin(model_holder.objective, space=model_holder.hyperparams, algo=algo, max_evals=max_evals)

# TODO provide way to utilize fmin return value
