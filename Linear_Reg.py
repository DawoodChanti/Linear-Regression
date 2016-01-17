"""
Linear Regression Problem
http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex2/ex2.html
"""

import numpy as np
import matplotlib.pyplot as plt

# Read and Plot the dataset.


# Open the File
xdata=open('ex2x.dat',mode='r')
ydata=open('ex2y.dat',mode='r')


# Read the Data in form of a list of string
xs=[line.rstrip('\n').replace('e+00','') for line in open('ex2x.dat')]
# Transform the list of strings into float
xs =np.array(  [float(xs[i]) for i in range(len(xs))])

# Same for the y'axis data
ys= [line.rstrip('\n').replace('e+00','') for line in open('ex2y.dat')]
ys =np.array( [float(ys[i]) for i in range(len(ys))])



"""
Supervised learning problem :  Linear Regression

In this problem, I'll implement linear regression using gradient descent. 
"""

# store the number of training examples
m = len(ys);

#Add a column of ones to x-data
x = np.zeros(shape=(m, 2))

# basically a straight line
for i in range(0, m):
    x[i][0] = 1
    x[i][1] = xs[i]


"""
So, basiclly :
theta_0 and theta+1 are the parameter we want to minimize and update each time.

cost function = 1/2m sum (h - y)^2
gradient = (1/m)*(sum(from 1 to m) (h_theta - y)). x     Updating Rule
theta:=theta - learning_rate* gradient

"""

# m denotes the number of examples here
def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta


# Exectution

n= 2 #number of features is n
numIterations= 1500
alpha = 0.07 # Learning rate
theta = np.zeros(n)   # Initialize theta_0 and theta_1 with zeros
theta = gradientDescent(x, ys, theta, alpha, m, numIterations)
print(theta)


# Build the Model that predict the house price
prediction=np.zeros(50)
for i in range(m):
    prediction[i] = theta[0] +  x[i][1]*theta[1]


# Now Plot the Data :
plt.plot(xs, ys, marker='o', linestyle='--', color='b', label='Real Data')
plt.plot(xs, prediction, marker='o', linestyle='-', color='red', label='Predicted Data')
plt.xlabel('Ages (years)')
plt.ylabel('Heights (meters)')
plt.title('Data Representation for 50 Training Example')
plt.legend()
plt.show()
plt.savefig("Data_Representation.png")






#Predict values for age 3.5 and 7
predict1 = theta[0] + theta[1]*3.5
print predict1
predict2 = theta[0] + theta[1]*7
print predict2






###############################
 #other way
 
 #  https://github.com/mcassiano/ai-hw3-machine-learning/blob/73b4156c2dfb40047a16338ddab3ff7c7bb9798c/Problem1_1.py
def calculate_cost(loss, m):
    '''
        Calculates the risk (cost?) of a given
        vector loss = (prediction - y)
    '''
    return np.sum(loss ** 2.0) / (2.0 * m)

def gradient_descent(x, y, theta, alpha, numIterations):
    '''
        Performs gradient descent (multivariate) on x and y
        variating theta with a learning rate alpha. Runs
        numIterations times.
    '''

    m = x.shape[0]
    xTrans = x.transpose()

    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient

    return theta

def normal_equation(x, y):
    '''
        Performs the Normal Equation approach.
        theta = (x^T . x)^-1 . x^T . y 
    '''
    I=np.identity(2)
    return np.dot(np.dot((x.T.dot(x))*I, x.T), y)



# Pow shape cost function Ploting

theta0_vals = np.linspace(-1.0, 1.0, 100)
theta1_vals = np.linspace(-1.0, 1.0, 100)

Z = np.zeros(shape=(theta0_vals.size, theta1_vals.size))

for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = np.zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2

        guess = np.dot(x, thetaT).flatten()
        loss = guess - ys

        Z[t1, t2] = calculate_cost(loss, m)

X, Y = np.meshgrid(theta0_vals, theta1_vals)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z)




# Selectiong the Best Learning Rate (its obvious from graph that it is 0.07)
alphas = [0.005, 0.001, 0.05, 0.07]
iterations_n = 1500
iterations = np.arange(iterations_n)
risk = np.zeros(shape = (iterations_n, len(alphas))).T

for alpha_i in range(0, len(alphas)):
    theta_sim = np.zeros(n)
    for iteration_n in iterations:
        theta_sim = gradient_descent(x, ys, theta_sim, alphas[alpha_i], iteration_n)
        prediction = np.dot(x, theta_sim)
        loss = prediction - ys
        risk[alpha_i][iteration_n] = calculate_cost(loss, m)

for alpha_i in range(0, len(alphas)):
    plt.plot(iterations, risk[alpha_i], label='Alpha: %10.2f' % alphas[alpha_i])
    plt.legend()
