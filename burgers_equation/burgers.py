"""use Ravindra_env"""

import deepxde as dde
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def gen_testdata():
    data = np.load("/home/hpcs_rnd/RAVINDRA_IISc/burgers_equation/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y

#defining pde
def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

#defining domain and initial and boundary conditions 
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160
)

#neural network model 
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

#optimization 
model.compile("adam", lr=1e-3)
model.train(iterations=15000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))

# Compute correlation
correlation, _ = pearsonr(y_true.flatten(), y_pred.flatten())
print("correlation coefficient:", correlation)

# Compute comparison metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared:", r2)

#ploting comparision with exact solution
data = np.load("/home/hpcs_rnd/RAVINDRA_IISc/burgers_equation/Burgers.npz")
t, x, exact = data["t"], data["x"], data["usol"].T

# Reshape y_true and y_pred for comparison
y_true_reshaped = y_true.reshape(len(t), len(x))
y_pred_reshaped = y_pred.reshape(len(t), len(x))

# Indices for initial and final time
initial_idx = np.argmin(np.abs(t - 0))
final_idx = np.argmin(np.abs(t - 0.99))

# Plotting
plt.figure(figsize=(14, 4))

# Initial time comparison
plt.subplot(1, 2, 1)
plt.plot(x, y_true_reshaped[initial_idx], label='Exact (Initial)')
plt.plot(x, y_pred_reshaped[initial_idx], 'r--', label='Predicted (Initial)')
plt.title('Initial Time Comparison')
plt.xlabel('Spatial Domain')
plt.ylabel('Solution')
plt.legend()

# Final time comparison
plt.subplot(1, 2, 2)
plt.plot(x, y_true_reshaped[final_idx], label='Exact (Final)')
plt.plot(x, y_pred_reshaped[final_idx], 'r--', label='Predicted (Final)')
plt.title('Final Time Comparison')
plt.xlabel('Spatial Domain')
plt.legend()

plt.tight_layout()
save_path = '/home/hpcs_rnd/RAVINDRA_IISc/burgers_equation/plot/comp'
plt.savefig(save_path)

#plotting heatmap of deifference btween pinns and exact
# Reshape y_true and y_pred for heatmap
y_true_reshaped = y_true.reshape(len(t), len(x))
y_pred_reshaped = y_pred.reshape(len(t), len(x))

# Calculate the difference
difference = (y_true_reshaped - y_pred_reshaped).T

# # Heatmap of the difference
# plt.figure(figsize=(8, 6))
# plt.imshow(difference, extent=[x.min(), x.max(), t.min(), t.max()], origin='lower', aspect='auto', cmap='RdBu')
# plt.colorbar(label='Difference')
# plt.title('Heatmap of the Difference between Exact and Predicted Solutions')
# plt.xlabel('Spatial Domain')
# plt.ylabel('Time')
# plt.show()

fig, ax = plt.subplots(figsize=(10, 5))

# Heatmap of the solution u(t, x)
cmap = cm.viridis  # Choose a colormap
heatmap = ax.imshow(difference, extent=[t.min(), t.max(), x.min(), x.max()], aspect='auto', cmap=cmap, origin='lower')
ax.set_xlabel('Time t')
ax.set_ylabel('Space x')
ax.set_title('Difference between Exact and Predicted Solutions')

# Adding a colorbar
cbar = fig.colorbar(heatmap, ax=ax)
cbar.set_label('Difference')

save_path = '/home/hpcs_rnd/RAVINDRA_IISc/burgers_equation/plot/diff'
plt.savefig(save_path)

fig, ax = plt.subplots(figsize=(10, 5))

# Heatmap of the solution u(t, x)
cmap = cm.viridis  # Choose a colormap
heatmap = ax.imshow(y_pred_reshaped.T, extent=[t.min(), t.max(), x.min(), x.max()], aspect='auto', cmap=cmap, origin='lower')
ax.set_xlabel('Time t')
ax.set_ylabel('Space x')
ax.set_title('Predicted u(t, x)')

# Adding a colorbar
cbar = fig.colorbar(heatmap, ax=ax)
cbar.set_label('Predicted u(t, x)')

save_path = '/home/hpcs_rnd/RAVINDRA_IISc/burgers_equation/plot/pred'
plt.savefig(save_path)

