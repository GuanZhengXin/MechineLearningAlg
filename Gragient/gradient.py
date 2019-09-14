import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(theta,eta,epsilon=1e-8):
    assert theta is not None,'theta is not None'
    assert eta is not None,'eat is not None'
    theta_history= []
    theta_history.append(theta)
    while True:
        gradient = Dj(theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_history.append(theta)
        if(abs(J(last_theta)-J(theta))<=epsilon):
            break
    theta_history = np.array(theta_history)
    return theta,theta_history

def Dj(theta):
    return 2*(theta-2.5)

def J(theta):
    return np.array((theta-2.5)**2-1)

x = np.linspace(-1,6,141)
y = J(x)
plt.plot(x,y)
theta,theta_history = gradient_descent(0,0.1)
print(theta)
plt.plot(theta_history,J(theta_history),color='r',marker='+')
plt.show()


