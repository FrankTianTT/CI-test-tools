import numpy as np
import math


def pendulum_derivatives(theta, omega, g=9.8, l=1):
    """
    \dot{\theta} = \omega
    \dot{\omega} = -\frac{g \sin\theta}{l}
    :param theta: angel of the pendulum
    :param omega: angular velocity of the pendulum
    :param g: gravitational acceleration
    :param l: length of the pendulum
    :return: derivative of angel, derivative of angular velocity
    """
    d_theta = omega
    d_omega = - np.sin(theta) * g / l
    return d_theta, d_omega


def simulate(sample_num, time_scale=0.1, simulate_length=100,
             theta_std=np.pi, omega_std=1,
             seed=623, noise_std=0.5):
    """
    Simulate single pendulum.
    :param sample_num:
    :param time_scale:
    :param simulate_length:
    :param theta_std:
    :param omega_std:
    :param seed:
    :param noise_std:
    :return: simulated data
    """
    np.random.seed(seed)
    batch_size = math.ceil(sample_num / simulate_length)

    data = np.zeros([simulate_length * batch_size, 4])
    theta = np.random.randn(batch_size, 1) * theta_std
    omega = np.random.randn(batch_size, 1) * omega_std

    for i in range(simulate_length):
        d_theta, d_omega = pendulum_derivatives(theta, omega)
        data[i * batch_size: (i + 1) * batch_size] = np.hstack((theta, omega, d_theta, d_omega))
        theta += d_theta * time_scale
        omega += d_omega * time_scale

    return data[:sample_num]


if __name__ == '__main__':
    data = simulate(3010)
    print(data[-1])
    print(data.shape)
