import matplotlib.pyplot as plt
import matplotlib as mpl
## %matplotlib inline
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import sympy as sy
sy.init_printing()




plt.clf()
######################## Subspace W ##############################
s = np.linspace(-1, 1, 10)
t = np.linspace(-1, 1, 10)
S, T = np.meshgrid(s, t)

# define our vectors
vec = np.array([[[0, 0, 0, 3, 6, 2]],
                [[0, 0, 0, 1, 2, 4]],
                [[0, 0, 0, 2, -2, 1]]])

X = vec[0,:,3] * S + vec[1,:,3] * T
Y = vec[0,:,4] * S + vec[1,:,4] * T
Z = vec[0,:,5] * S + vec[1,:,5] * T

fig = plt.figure(figsize = (7, 7))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, Z, linewidth = 1.5, alpha = .3)

############################# x1 and x2 ##############################
colors = ['r','b','g']
s = ['$a_1$', '$a_2$', '$a_3$']
for i in range(vec.shape[0]):
    X,Y,Z,U,V,W = zip(*vec[i,:,:])
    ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False,
              color = colors[i], alpha = .6,arrow_length_ratio = .08, pivot = 'tail',
              linestyles = 'solid',linewidths = 3)
    ax.text(vec[i,:,3][0], vec[i,:,4][0], vec[i,:,5][0], s = s[i], size = 15)

ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.show()


"""


# define norm 
def norm(v):
    
    #    Compute the 2-norm of given vector.
    #   Assume we use Frobenious norm (Euclidean norm)
    #   in: v - a batch of vector of shape [N, D] 
    #   out: a batch of computed 2-norm of shape [N, D]
    
    ############################### TODO ###################################




# our vectors that span the 3D space.
a1 = np.array([3, 6, 2])
a2 = np.array([1, 2, 4])
a3 = np.array([2, -2, 1])

# set the a1 as the q_tilde_1
q_tilde1 = #### TODO ####

# normalize q_1
q1 = #### TODO ####

# yield q_tilde2 that is orthogonal to q_1
q_tilde2 = #### TODO ####

# normalize q_2
q2 = #### TODO ####

# yield q_tilde3 that is orthogonal to q_1 and q_2
q_tilde3 = #### TODO ####

# normalize q_3
q3 = #### TODO ####






plt.clf()
######################## Subspace W ##############################

s = np.linspace(-1, 1, 10)
t = np.linspace(-1, 1, 10)
S, T = np.meshgrid(s, t)

a1, q_tilde1 = np.array([3, 6, 2]), np.array([3, 6, 2])
a2 = np.array([1, 2, 4])
a3 = np.array([2, -2, 1])

X = a1[0] * S + a2[0] * T
Y = a1[1] * S + a2[1] * T
Z = a1[2] * S + a2[2] * T

fig = plt.figure(figsize = (7, 7))
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, Z, linewidth = 1.5, alpha = .3)

############################# x1, x2, v2, alpha*v1 ##############################

vec = np.array([[0, 0, 0, a1[0], a1[1], a1[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'red', alpha = .6, arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'dashed',linewidths = 1)

vec = np.array([[0, 0, 0, a2[0], a2[1], a2[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'blue', alpha = .6, arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'dashed',linewidths = 1)

vec = np.array([[0, 0, 0, a3[0], a3[1], a3[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'green', alpha = .6, arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'dashed',linewidths = 1)

vec = np.array([[0, 0, 0, q1[0], q1[1], q1[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'black', alpha = 1.0, arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)

vec = np.array([[0, 0, 0, q_tilde2[0], q_tilde2[1], q_tilde2[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'black', alpha = 1.0, arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'dashed',linewidths = 1)

vec = np.array([[0, 0, 0, q2[0], q2[1], q2[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'black', alpha = 1.0, arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'dashed',linewidths = 3)


vec = np.array([[0, 0, 0, q_tilde3[0], q_tilde3[1], q_tilde3[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'gray', alpha = 0.3, arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'dashed',linewidths = 1)

vec = np.array([[0, 0, 0, q3[0], q3[1], q3[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'black', alpha = 1.0, arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)


ax.text(a1[0] * 1.1, a1[1] * 1.1, a1[2] * 1.1, '$\mathbf{a}_1 = \mathbf{\~{q}}_1 $', size = 10)
ax.text(a2[0] * 1.1, a2[1] * 1.1, a2[2] * 1.1, '$\mathbf{a}_2$', size = 10)
ax.text(a3[0] * 1.1, a3[1] * 1.1, a3[2] * 1.1, '$\mathbf{a}_3$', size = 10)
ax.text(q1[0] * 1.1, q1[1] * 1.1, q1[2] * 1.1, '$\mathbf{q}_1$', size = 15)

ax.text(q_tilde2[0] * 1.1, q_tilde2[1] * 1.1, q_tilde2[2] * 1.1, '$\mathbf{\~{q}}_2$', size = 10)
ax.text(q2[0] * 1.1, q2[1] * 1.1, q2[2] * 1.1, '$\mathbf{q}_2$', size = 15)

ax.text(q_tilde3[0] * 1.1, q_tilde3[1] * 1.1, q_tilde3[2] * 1.1, '$\mathbf{\~{q}}_3$', size = 10)
ax.text(q3[0] * 1.1, q3[1] * 1.1, q3[2] * 1.1, '$\mathbf{q}_3$', size = 15)


ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')

plt.show()





print(np.allclose(q1 @ q2, 0.))
print(np.allclose(q3 @ q2, 0.))
print(np.allclose(q3 @ q1, 0.))




def gram_schmidt(a):
    
    # in: a - length k list of n-dim np.arrays.
    # out: q - length k list of n-dim orthonormal np.arrays. 
    
    ############################### TODO ###################################
    



a = np.vstack([a1, a2, a3])
q = gram_schmidt(a)
#Test orthonormality
print('Norm of q[0] :', (sum(q[0]**2))**0.5)
print('Inner product of q[0] and q[1] :', q[0] @ q[1])
print('Inner product of q[0] and q[2] :', q[0] @ q[2])
print('Norm of q[1] :', (sum(q[1]**2))**0.5)
print('Inner product of q[1] and q[2] :', q[1] @ q[2])
print('Norm of q[2] :', (sum(q[2]**2))**0.5)


"""