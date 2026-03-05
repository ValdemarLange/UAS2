import numpy as np
from typing import Callable, Tuple
from icecream import ic
from scipy.stats import chi2
import matplotlib.pyplot as plt



class LevenbergMarquardt():
    def __init__(self, 
                 function: Callable[[np.ndarray], np.ndarray], 
                 initial_param: np.ndarray, 
                 damping: float = 100
                 ) -> None:
        self.parameters_to_optimize = initial_param == initial_param
        self.damping = damping
        self.func = function
        self.param = initial_param
        self.residual_error = np.linalg.norm(self.func(self.param))


    def jacobian(self, 
                 param: np.ndarray, 
                 func: Callable[[np.ndarray], np.ndarray]
                 ) -> Tuple[np.ndarray, np.ndarray]: 
        e = 0.00001
        delta = np.zeros(param.shape)

        # Calculate the function values at the given pose
        projected_points = func(param)

        # Calculate jacobian by perturbing the pose prior 
        # to calculating the function values.
        j = np.zeros((projected_points.shape[1], param.shape[0]))
        for k in range(param.shape[0]):
            delta_k = delta.copy()
            delta_k[k] = e
            param_temp = param + delta_k.transpose()
            func_value = func(param_temp)
            j[:, k] = (func_value - projected_points) / e

        # Limit the jacobian to the parameters that should be optimized.
        j = j[:, self.parameters_to_optimize]

        return (projected_points, j)


    def iterate(self) -> None:
        # Get projection errors and the associated jacobian
        self.projection_errors, j = self.jacobian(self.param, self.func)

        # Levenberg Marquard update rule
        self.coefficient_covariance_matrix = j.transpose() @ j
        t2 = np.diag(np.diag(self.coefficient_covariance_matrix)) * self.damping
        t3 = self.coefficient_covariance_matrix + t2
        param_update = np.linalg.inv(t3) @ j.transpose() @ self.projection_errors.transpose()

        # Unpack to full solution
        dx = np.zeros((self.param.shape[0], 1))
        dx[self.parameters_to_optimize] = param_update
        updated_x = self.param - dx.reshape((-1))
        updated_residual_error = np.linalg.norm(self.func(updated_x))

        if self.residual_error < updated_residual_error:
            # Squared error was increased, reject update and increase damping
            self.damping = self.damping * 10
        else:
            # Squared error was reduced, accept update and decrease damping
            self.param = updated_x
            self.damping = self.damping / 3
            self.residual_error = updated_residual_error

        return
    
    def estimate_uncertainties(self, p = 0.99):
        self.squared_residual_error = self.residual_error**2
        number_of_observations = self.projection_errors.size
        number_of_parameters = self.parameters_to_optimize.size
        # https://www.youtube.com/watch?v=3IgIToOV2Wk at 4:39
        sigma_hat_squared = self.squared_residual_error / (number_of_observations - number_of_parameters)
        ic(sigma_hat_squared)

        # Determine how many standard deviations we should go out
        # to cover a given probability (p).
        # TODO: I am unsure if it should be split into these two cases (one_dim vs multi_dim)
        self.scale_one_dim = chi2.ppf(p, 1)
        self.scale_multi_dim = chi2.ppf(p, self.param.size)
        # self.scale_multi_dim = self.scale_one_dim

        # Equation 15.4.15 from Numerical Recipes in C 2002
        self.param_uncert = self.scale_one_dim * np.sqrt(sigma_hat_squared) / np.sqrt(np.diag(self.coefficient_covariance_matrix))

        # Equation on page 660 in Numerical Recipes in C 2002
        self.goodness_of_fit = 1 - chi2.cdf(self.residual_error**2, self.projection_errors.size)
    
        # Build matrix with uncertainties for independent parameters
        # Equation 15.4.15 from Numerical Recipes in C 2002
        delta = np.zeros(self.param.shape)
        self.independent_uncertainties = np.zeros((self.param.size, self.param.size))
        for k in range(self.param.size):
            delta_k = delta.copy()
            delta_k[k] = 1
            vector = self.param_uncert * delta_k
            self.independent_uncertainties[k, :] = vector
        ic(self.independent_uncertainties)

        # Build matrix with uncertainties for combined parameters
        # Based on equation 15.4.18 in Numericap Recipes in C 2002
        u, s, vh = np.linalg.svd(np.linalg.inv(self.coefficient_covariance_matrix))
        self.combination_uncert = self.scale_multi_dim * np.sqrt(s * sigma_hat_squared)
        self.combined_uncertainties = np.zeros((self.param.size, self.param.size))
        for k in range(self.param.size):
            vector = self.scale_multi_dim * vh[k] * np.sqrt(s[k] * sigma_hat_squared)
            self.combined_uncertainties[k, :] = vector
        ic(self.combined_uncertainties)

        



def main():
    np.random.seed(1)
    n = 400
    measurement_error = 15
    def model1(xvals, param):
        return param[0] + xvals * param[1] + xvals * xvals * param[2]
    def model2(xvals, param):
        return param[0] + np.exp(0.01*xvals) * param[1] + np.exp(0.1 * xvals) * param[2]
    xvals = (np.random.rand(1, n) - 0.5) * 100
    yvals = model1(xvals, [2, 1, 0.01]) + np.random.randn(1, n) * measurement_error
    model = model2
    def test(param):
        predictions = model(xvals, param)
        # errors = (yvals - predictions) / measurement_error
        errors = (yvals - predictions)
        return errors
    
    lm = LevenbergMarquardt(test, np.array([1, 10, 100]))
    for k in range(20):
        lm.iterate()
    lm.estimate_uncertainties(p = 0.999)
    ic(lm.param)
    ic(lm.scale_one_dim)
    ic(lm.scale_multi_dim)
    ic(lm.squared_residual_error)
    ic(lm.goodness_of_fit)

    # Testing the uncertainty of estimates by perturbing the estimated
    # parameters by the calculated uncertainties. If done correctly, this
    # should give the same error norm in all three cases.
    # This seems to hold very well!
    print("independent uncertainties")
    ic(lm.param_uncert)
    ic(lm.independent_uncertainties)
    temp0r = np.linalg.norm(test(lm.param))
    temp1a = np.linalg.norm(test(lm.param + lm.independent_uncertainties[0]))
    temp1b = np.linalg.norm(test(lm.param + lm.independent_uncertainties[1]))
    temp1c = np.linalg.norm(test(lm.param + lm.independent_uncertainties[2]))
    ic(temp0r)
    ic(temp1a)
    ic(temp1b)
    ic(temp1c)


    # Finally try to invert the matrix first and then do the SVD
    # This looks like the easiest way to interpret the 
    # uncertainties. It also means that the most important directions 
    # of uncertainties are listed first.
    print("combined uncertainties")
    ic(lm.combination_uncert)
    ic(lm.combined_uncertainties)
    val = np.abs(lm.combined_uncertainties)
    ic(np.max(val, axis=0))
    temp1vec = np.linalg.norm(test(lm.param + lm.combined_uncertainties[0]))
    temp2vec = np.linalg.norm(test(lm.param + lm.combined_uncertainties[1]))
    temp3vec = np.linalg.norm(test(lm.param + lm.combined_uncertainties[2]))
    ic(temp0r)
    ic(temp1vec)
    ic(temp2vec)
    ic(temp3vec)


    x_vector = np.linspace(-50, 50, 100)

    fig, ax = plt.subplots(1, 1)
    ax.plot(xvals, yvals, 'o')
    ax.plot(x_vector, model(x_vector, lm.param), 'k-')
    ax.plot(x_vector, model(x_vector, lm.param + lm.independent_uncertainties[0]), 'r-')
    ax.plot(x_vector, model(x_vector, lm.param - lm.independent_uncertainties[0]), 'r:')
    ax.plot(x_vector, model(x_vector, lm.param + lm.independent_uncertainties[1]), 'g-')
    ax.plot(x_vector, model(x_vector, lm.param - lm.independent_uncertainties[1]), 'g:')
    ax.plot(x_vector, model(x_vector, lm.param + lm.independent_uncertainties[2]), 'b-')
    ax.plot(x_vector, model(x_vector, lm.param - lm.independent_uncertainties[2]), 'b:')
    plt.savefig("mygraph_individual.png", 
                dpi = 300)
    plt.clf()
    ax = plt.axes()
    ax.plot(xvals, yvals, 'o')
    ax.plot(x_vector, model(x_vector, lm.param), 'k-')
    ax.plot(x_vector, model(x_vector, lm.param + lm.combined_uncertainties[0]), 'r-')
    ax.plot(x_vector, model(x_vector, lm.param - lm.combined_uncertainties[0]), 'r:')
    ax.plot(x_vector, model(x_vector, lm.param + lm.combined_uncertainties[1]), 'g-')
    ax.plot(x_vector, model(x_vector, lm.param - lm.combined_uncertainties[1]), 'g:')
    ax.plot(x_vector, model(x_vector, lm.param + lm.combined_uncertainties[2]), 'b-')
    ax.plot(x_vector, model(x_vector, lm.param - lm.combined_uncertainties[2]), 'b:')
    plt.savefig("mygraph_combinations.png", 
                dpi = 300)



np.set_printoptions(formatter={'float': lambda x: "{0:7.3f}".format(x)})
main()
