import numpy as np
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')


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


def cci_test(x, y, z,
             train_ratio=2 / 3,
             model_class=None,
             k=1,
             seed=623):
    np.random.seed(seed)
    assert len(x) == len(y) == len(z)
    sample_num = len(x)
    train_num = int(sample_num * train_ratio)
    permutation = np.random.permutation(sample_num)
    train_index = permutation[:train_num]
    test_index = permutation[train_num:]

    train_data, bias_train_data, train_label = nearest_neighbor_bootstrap(x[train_index], y[train_index],
                                                                          z[train_index], k)
    test_data, bias_test_data, test_label = nearest_neighbor_bootstrap(x[test_index], y[test_index], z[test_index], k)

    model = XGBClassifier()
    model.fit(bias_train_data, train_label)
    bias_score = model.score(bias_test_data, test_label)

    model = XGBClassifier()
    model.fit(train_data, train_label)
    score = model.score(test_data, test_label)

    print(score - bias_score)
    pass


if __name__ == '__main__':
    from CI_test.data_generation.simple_sin_cos import gen_cond_ind_data, gen_non_cond_ind_data

    x, y, z = gen_non_cond_ind_data(3000)
    cci_test(x, y, z)
