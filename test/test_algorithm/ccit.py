import unittest
from CI_test.data_generation.simple_cos import gen_cond_ind_data, gen_non_cond_ind_data
from CI_test.data_generation.single_pendulum import simulate
from CI_test.algorithm.ccit import cci_test, xgb_cross_validate
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_con_ind_simple_cos(self):
        p_values = []
        best_params = xgb_cross_validate(*gen_cond_ind_data(3000))
        for i in range(5):
            p_value = cci_test(*gen_cond_ind_data(3000, seed=i), bootstrap=False, seed=i, best_params=best_params)
            p_values.append(p_value)
        p_values = np.array(p_values)
        self.assertTrue((p_values > 0.05).all())

    def test_non_con_ind_simple_cos(self):
        p_values = []
        best_params = xgb_cross_validate(*gen_non_cond_ind_data(3000))
        for i in range(5):
            p_value = cci_test(*gen_non_cond_ind_data(3000, seed=i), bootstrap=False, seed=i, best_params=best_params)
            p_values.append(p_value)
        p_values = np.array(p_values)
        self.assertTrue((p_values < 0.05).all())

    def test_on_single_pendulum(self):
        data = simulate(3000)
        theta, omega, d_theta, d_omega = [data[:, i].reshape(-1, 1) for i in range(4)]
        p_value1 = cci_test(theta, d_omega, omega)
        p_value2 = cci_test(theta, d_theta, omega)
        p_value3 = cci_test(omega, d_theta, theta)
        p_value4 = cci_test(omega, d_omega, theta)
        self.assertTrue(p_value1 < 0.05 and p_value2 > 0.05 and p_value3 < 0.05 and p_value4 > 0.05)


if __name__ == '__main__':
    unittest.main()
