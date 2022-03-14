import numpy as np


def get_unit_vector(dim):
    vector = np.random.randn(dim, 1)
    return vector / np.sqrt(np.sum(np.square(vector)))


def gen_cond_ind_data(sample_num, z_dim=10, seed=623, noise_std=0.25):
    """
    Method for generating conditional independent data.
    :return: x, y and z, with relationship x <- z -> y.
    """""
    np.random.seed(seed)
    z = np.random.randn(sample_num, z_dim)

    param_a = get_unit_vector(z_dim)
    param_b = get_unit_vector(z_dim)
    noise_x = np.random.randn(sample_num, 1) * noise_std
    noise_y = np.random.randn(sample_num, 1) * noise_std

    x = np.cos(np.matmul(z, param_a) + noise_x)
    y = np.cos(np.matmul(z, param_b) + noise_y)

    return x, y, z


def gen_non_cond_ind_data(sample_num, z_dim=10, seed=623, noise_std=0.25, c_scale=0.2):
    """
    Method for generating conditional dependent data.
    :return: x, y and z, with relationship x <- z -> y and x -> y.
    """""
    np.random.seed(seed)
    z = np.random.randn(sample_num, z_dim)

    param_a = get_unit_vector(z_dim)
    param_b = get_unit_vector(z_dim)
    param_c = np.random.random() * c_scale
    noise_x = np.random.randn(sample_num, 1) * noise_std
    noise_y = np.random.randn(sample_num, 1) * noise_std

    x = np.cos(np.matmul(z, param_a) + noise_x)
    y = np.cos(np.matmul(z, param_b) + param_c * x + noise_y)

    return x, y, z


if __name__ == '__main__':
    x, y, z = gen_cond_ind_data(100)
    print(x.shape)
    print(z.shape)
