import warnings

warnings.filterwarnings('ignore')

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from math import erfc
from multiprocessing import Pool
from functools import partial


def nearest_neighbor_bootstrap(x, y, z,
                               k=1):
    sample_num = len(x)
    shuffle = np.random.permutation(2 * sample_num)
    neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", metric="l2").fit(z)
    distances, indices = neighbors.kneighbors(z)
    y_prime = np.zeros(y.shape)

    for i in range(sample_num):
        y_prime[i] = y[indices[i][k]]

    origin_data = np.concatenate([x, y, z], axis=-1)
    sampled_data = np.concatenate([x, y_prime, z], axis=-1)
    bias_origin_data = np.concatenate([y, z], axis=-1)
    bias_sampled_data = np.concatenate([y_prime, z], axis=-1)
    data = np.concatenate([origin_data, sampled_data], axis=0)
    bias_data = np.concatenate([bias_origin_data, bias_sampled_data], axis=0)
    labels = np.concatenate([np.ones(sample_num), np.zeros(sample_num)], axis=0)
    return data[shuffle], bias_data[shuffle], labels[shuffle]


def xgb_cross_validate(x, y, z,
                       k=1,
                       max_depth=None, n_estimators=None, colsample_bytree=None,
                       default_xgb_params=None,
                       cv_n_fold=5,
                       n_thread=8,
                       verbose=False):
    if max_depth is None:
        max_depth = [6, 10, 13]
    if n_estimators is None:
        n_estimators = [100, 200, 300]
    if colsample_bytree is None:
        colsample_bytree = [0.4, 0.8]
    if default_xgb_params is None:
        default_xgb_params = dict(eval_metric="logloss",
                                  use_label_encoder=False,
                                  learning_rate=0.02,
                                  min_child_weight=1,
                                  gamma=0,
                                  subsample=0.8,
                                  objective='binary:logistic',
                                  scale_pos_weight=1,
                                  seed=623)
    data, _, label = nearest_neighbor_bootstrap(x, y, z, k)

    parameters = {'max_depth': max_depth, 'n_estimators': n_estimators, "colsample_bytree": colsample_bytree}
    xgb = XGBClassifier(**default_xgb_params)
    clf = GridSearchCV(xgb, parameters, cv=cv_n_fold, verbose=verbose, n_jobs=n_thread)
    clf.fit(data, label)
    return {**clf.best_params_, **default_xgb_params}


def xgb_train_and_eval(train_data, train_label,
                       test_data, test_label,
                       best_params):
    model = XGBClassifier(**best_params)
    model.fit(train_data, train_label)
    pred_prob = model.predict_proba(test_data)
    pred_exact = model.predict(test_data)
    acc = accuracy_score(test_label, pred_exact)
    auc = roc_auc_score(test_label, pred_prob[:, 1])
    return acc, auc


def xgb_out(x, y, z, best_params,
            train_ratio=2 / 3,
            k=1,
            threshold=0.03,
            bootstrap=False):
    sample_num = len(x)
    train_num = int(sample_num * train_ratio)
    if bootstrap:
        index = np.random.choice(sample_num, size=sample_num, replace=True)
    else:
        index = np.random.permutation(sample_num)
    train_index = index[:train_num]
    test_index = index[train_num:]
    train_data, bias_train_data, train_label = nearest_neighbor_bootstrap(x[train_index], y[train_index],
                                                                          z[train_index], k)
    test_data, bias_test_data, test_label = nearest_neighbor_bootstrap(x[test_index], y[test_index], z[test_index], k)

    acc, auc = xgb_train_and_eval(train_data, train_label, test_data, test_label, best_params)
    bias_acc, bias_auc = xgb_train_and_eval(bias_train_data, train_label, bias_test_data, test_label, best_params)

    if auc > bias_auc + threshold:
        return [0.0, auc - bias_auc, bias_auc - 0.5, acc - bias_acc, bias_acc - 0.5]
    else:
        return [1.0, auc - bias_auc, bias_auc - 0.5, acc - bias_acc, bias_acc - 0.5]


def pvalue(x, sigma):
    return 0.5 * erfc(x / (sigma * np.sqrt(2)))


def cci_test(x, y, z,
             train_ratio=2 / 3,
             max_depth=None, n_estimators=None, colsample_bytree=None,
best_params=None,
             cv_n_fold=5,
             k=1,
             threshold=0.03,
             num_iter=20,
             n_thread=8,
             bootstrap=False,
             seed=623,
             verbose=False):
    if max_depth is None:
        max_depth = [6, 10, 13]
    if n_estimators is None:
        n_estimators = [100, 200, 300]
    if colsample_bytree is None:
        colsample_bytree = [0.4, 0.8]

    np.random.seed(seed)
    assert len(x) == len(y) == len(z)
    sample_num = len(x)

    if best_params is None:
        best_params = xgb_cross_validate(x, y, z, k,
                                         max_depth=max_depth,
                                         n_estimators=n_estimators,
                                         colsample_bytree=colsample_bytree,
                                         cv_n_fold=cv_n_fold,
                                         n_thread=n_thread,
                                         verbose=verbose)
    if verbose:
        print(best_params)

    xgb_out_kwargs = dict(x=x, y=y, z=z, k=k,
                          best_params=best_params, train_ratio=train_ratio, threshold=threshold, bootstrap=bootstrap)
    if bootstrap:
        xgb_result = []
        pool = Pool(processes=n_thread)
        for i in range(num_iter):
            pool.apply_async(xgb_out, kwds=xgb_out_kwargs, callback=xgb_result.append)
        pool.close()
        pool.join()
    else:
        xgb_result = [xgb_out(**xgb_out_kwargs)]

    xgb_result = np.array(xgb_result)

    relative_acc = np.mean(xgb_result[:, 3])
    p_value = pvalue(relative_acc, 1 / np.sqrt(sample_num))

    return p_value


if __name__ == '__main__':
    from CI_test.data_generation.simple_cos import gen_cond_ind_data, gen_non_cond_ind_data

    pvals = []
    best_params = xgb_cross_validate(*gen_non_cond_ind_data(3000))
    print(best_params)
    for i in range(5):
        pval = cci_test(*gen_non_cond_ind_data(3000, seed=i), bootstrap=False, seed=i, best_params=best_params)
        pvals.append(pval)
    print(pvals)
    # [3.109724167965409e-07, 0.001183439954763605, 0.009256195044654905, 0.00022794000260280948, 0.0030849496602720593]

    pvals = []
    best_params = xgb_cross_validate(*gen_cond_ind_data(3000))
    print(best_params)
    for i in range(5):
        pval = cci_test(*gen_cond_ind_data(3000, seed=i), bootstrap=False, seed=i, best_params=best_params)
        pvals.append(pval)
    print(pvals)
    # [0.39209561470080945, 0.5218400480422483, 0.7356137583414126, 0.4455428101086095, 0.21354017120091398]
