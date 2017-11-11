from hyperopt import fmin, tpe, Trials
from hyperopt.mongoexp import MongoTrials


def find_best_model(model_holder, exp_key, algo=tpe.suggest, max_evals=10, mongodb_url=None):
    if mongodb_url is not None:
        trials = MongoTrials(mongodb_url, exp_key=exp_key)
    else:
        trials = Trials(exp_key=exp_key)

    fmin(model_holder.objective, space=model_holder.hyperparams, trials=trials, algo=algo, max_evals=max_evals)
    return trials
