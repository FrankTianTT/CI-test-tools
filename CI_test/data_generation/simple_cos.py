import numpy as np


def get_unit_array(dim1, dim2):
    array = np.random.rand(dim1, dim2)
    for i in range(dim2):
        array[:, i] = array[:, i] / np.linalg.norm(array[:, i])
    return array


def gen_cond_ind_data(sample_num, x_dim=1, y_dim=1, z_dim=10,
                      freq=1.0,
                      seed=623, noise_std=0.5):
    """
    Method for generating conditional independent data.
    :param sample_num:
    :param x_dim:
    :param y_dim:
    :param z_dim:
    :param freq:
    :param seed:
    :param noise_std:
    :return: x, y and z, with relationship x <- z -> y.
    """
    np.random.seed(seed)
    z = np.random.randn(sample_num, z_dim) + 1

    param_x = get_unit_array(z_dim, x_dim)
    param_y = get_unit_array(z_dim, y_dim)
    noise_x = np.random.randn(sample_num, x_dim) * noise_std
    noise_y = np.random.randn(sample_num, y_dim) * noise_std

    x = np.cos(freq * (np.matmul(z, param_x) + noise_x))
    y = np.cos(freq * (np.matmul(z, param_y) + noise_y))

    return x, y, z


def gen_non_cond_ind_data(sample_num,
                          x_dim=1, y_dim=1, z_dim=10,
                          freq=1.0,
                          seed=623, noise_std=0.5, dependent_scale=2):
    """
    Method for generating conditional dependent data.
    :param sample_num:
    :param x_dim:
    :param y_dim:
    :param z_dim:
    :param seed:
    :param noise_std:
    :param dependent_scale:
    :return: x, y and z, with relationship x <- z -> y and x -> y.
    """
    np.random.seed(seed)
    z = np.random.randn(sample_num, z_dim) + 1

    param_x = get_unit_array(z_dim, x_dim)
    param_y = get_unit_array(z_dim, y_dim)
    param_xy = get_unit_array(x_dim, y_dim) * dependent_scale
    noise_x = np.random.randn(sample_num, 1) * noise_std
    noise_y = np.random.randn(sample_num, 1) * noise_std

    x = np.cos(freq * (np.matmul(z, param_x) + noise_x))
    y = np.cos(freq * (np.matmul(z, param_y) + np.matmul(x, param_xy) + noise_y))

    return x, y, z


if __name__ == '__main__':
    x, y, z = gen_cond_ind_data(10)
    print(x.shape)
    print(z.shape)
