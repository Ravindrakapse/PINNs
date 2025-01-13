"""use Ravindra_env"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt


x_min, y_min, t_min = 0,0,0
x_max, y_max, t_max = 1,1,2
lr = 1e-4
iterations = 30000
data_path_name = "/home/hpcs_rnd/RAVINDRA_IISc/advection_equation/advection1_2d.npz"
input_num = 3
net_node = 100
layers = 5
num_domain=2540 
num_boundary=100 
num_initial=160 
num_test=2540 

#defining 2d advection equation
def pde(x, y):
    
    dy_t = dde.grad.jacobian(y, x, i=0, j=2) 
    dy_x = dde.grad.jacobian(y, x, i=0, j=0) 
    dy_y = dde.grad.jacobian(y, x, i=0, j=1) 

    return dy_x+dy_y+dy_t

# Initial and boundary conditions:
def initial_condition(x):
    return x[:,0:1] + x[:,1:2] 

#defining domain and initial and boundary conditions
geom = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])

timedomain = dde.geometry.TimeDomain(t_min, t_max)

geomtime = dde.geometry.GeometryXTime(geom, timedomain)

ic = dde.icbc.IC(
    
    geomtime,
    initial_condition,  
    lambda _, on_initial: on_initial,  
)

data = dde.data.TimePDE(
    geomtime, 
    pde, 
    [ic], 
    num_domain=num_domain, 
    num_boundary=num_boundary, 
    num_initial=num_initial,
    num_test=num_test, 
)
#neural network model
net = dde.nn.FNN([input_num] + [net_node] * layers + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

#optimization---> Adam
model.compile("adam", lr=lr)
model.train(iterations=iterations) 

#optimization---> L-BFGS
model.compile('L-BFGS')
losshistory, train_state = model.train()

#generating data for text
def gen_testdata(path_name): 
    
    data = np.load("/home/hpcs_rnd/RAVINDRA_IISc/advection_equation/advection1_2d.npz")
    
    t, x, y, exact = data["t"], data["x1"],data["x2"], data["usol"]
    
    xx, yy, tt = np.meshgrid(x, y, t) 
    X = np.vstack((np.ravel(xx), np.ravel(yy), np.ravel(tt))).T 
    y = exact.flatten()[:, None]
    return X, y

#saving plots and data
dde.saveplot(losshistory, train_state, issave=True, isplot=True) 
X, y_true = gen_testdata(data_path_name) 
y_pred = model.predict(X) 

#comparision matrics
mse = dde.metrics.mean_squared_error(y_true, y_pred)
print("Mean Squared Error:", mse)

individual_mses = [(true_val - pred_val) ** 2 for true_val, pred_val in zip(y_true, y_pred)]
mse_variance = sum([(mse - individual_mse) ** 2 for individual_mse in individual_mses]) / len(individual_mses)
print("Mean Squared Error Variance:", mse_variance)

#plotting heatmap
data = np.load('advection_equation/advection1_2d.npz')
x,y,t,exact = data['x1'],data['x2'],data['t'],data['usol']
X, Y = np.meshgrid(x.flatten(), y.flatten())

y_pred = y_pred.reshape(100,100,100)
fig, axs = plt.subplots(1, 2, figsize=(16, 6))


c = axs[0].contourf(X, Y, y_pred[0], cmap='viridis')
axs[0].set_title(f'Timestamp: {t[0][0]:.2f}')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
plt.colorbar(c, ax=axs[0], label='Color Intensity')


c = axs[1].contourf(X, Y, y_pred[-1], cmap='viridis')
axs[1].set_title(f'Timestamp: {t[-1][0]:.2f}')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
plt.colorbar(c, ax=axs[1], label='Color Intensity') 

plt.tight_layout()
save_path = '/home/hpcs_rnd/RAVINDRA_IISc/advection_equation/plot/pred'
plt.savefig(save_path)