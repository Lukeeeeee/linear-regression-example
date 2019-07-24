from baconian.algo.dynamics.linear_dynamics_model import LinearRegressionDynamicsModel
from baconian.core.core import EnvSpec
from baconian.common.spaces.box import Box
import numpy as np

# since baconian is designed for model-based RL, and LinearRegressionDynamicsModel is not a general linear regression,
# instead, it is designed to approximate a state transition, which means it takes the state and action and input, and
# output the new state of next time step. Here I encapsulated the details and provide a function served as general
# purpose linear regression.

# define parameters here
input_dim = 5
output_dim = 1


def fit(x, y, linear_model):
    linear_model._linear_model.fit(X=x, y=y)


def setup_model(input_dim, output_dim):
    env_spec = EnvSpec(obs_space=Box(low=-1.0, high=1.0, shape=output_dim),
                       action_space=Box(low=-1.0, high=1.0, shape=input_dim))
    a = LinearRegressionDynamicsModel(env_spec=env_spec)
    return a


def print_model_params(linear_model):
    print("coefficients: ", linear_model._linear_model.coef_)
    print("bias: ", linear_model._linear_model.intercept_)


linear_model = setup_model(input_dim, output_dim)

# use a simple sum operation linear model
x_train_data = np.random.random([10, input_dim])
y_train_data = np.expand_dims(np.sum(x_train_data, axis=-1), axis=-1)

# fit
print("fit with data")
print("x: ", x_train_data)
print("y: ", y_train_data)

fit(x_train_data, y_train_data, linear_model)
# print the parameters, you should see the
print_model_params(linear_model)
