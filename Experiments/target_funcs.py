import numpy as np
import math

def f_quad(X):

    return .02*np.power(X[0],2) + .005*np.power(X[1],2)

def grad_f_quad(X):

    return np.array([.04*X[0], .01*X[1]])


A = np.random.normal(0,1,(100,500))
b = np.random.normal(0,9, (100))

step_least_squares = 1.0/(np.max(np.linalg.eig(A.dot(A.transpose()))[0]) )
lambdaparam = 0.0001




A_logistic = np.random.normal(0,1,(500,100))
x0 = np.random.normal(0,.01, (100))
yis = np.array([np.random.binomial(1, 1.0/(1+np.exp(-A_logistic[i,:].dot(x0)))) for i in range(500)] )


# X must be 100 dimensional
def logistic_regression(X):
	def helper(u):
			return np.log(1+np.exp(u))

	return sum([-yis[i]*A_logistic[i,:].dot(X)+ np.log(1+ np.exp(A_logistic[i,:].dot(X))) for i in range(500)])
	
#return np.sum(-yis.dot(A_logistic).dot(X) + map(helper, A_logistic.dot(X)))




def grad_logistic_regression(X):
	#grad = np.zeros((100)) #dim 100
	grad = -A_logistic.transpose().dot(yis) #dim 100
	dot_products = A_logistic.dot(X) #dim 500
	def helper(y):
		return np.exp(y)/(1.0+ np.exp(y))
	partial_result = map(helper, dot_products) #dim 500
	grad += A_logistic.transpose().dot(partial_result)


	#for i in range(500):
	#	grad += -yis[i]*A_logistic[i,:]
	#	grad += np.exp(A_logistic[i,:].dot(X))/(1.0+np.exp(A_logistic[i,:].dot(X)))*A_logistic[i,:]
	return grad



def least_squares(X):
	return 0.5*np.linalg.norm(np.dot(A,X) - b)**2 

# Assuming f(x) = g(x) + h(x)
# x - argmin( || z - (x - s\nabla g(x) ||^2)/(2s) + h(z))
def grad_least_squares(X):
		result = A.transpose().dot(A.dot(X)-b)
		return result


def lasso_fat_design(X):
		return 0.5*np.linalg.norm(np.dot(A,X) - b)**2  + lambdaparam*np.sum(np.abs(X))


def soft_treshold(x, threshold = lambdaparam):
		#if threshold < 0:
		#		return
		#if x >= threshold:
		#		return x - threshold
		#elif x <= -threshold:
		#		return x + threshold
		#else:
		#		return 0		

		if threshold < 0:
				return
		if x >= 0 :
				return threshold
		elif x <= 0:
				return -threshold
		else:
				return 0



def lasso_fat_design_gradient(X):
	least_squares_gradient = grad_least_squares(X)
	#U = X - step_least_squares*least_squares_gradient 
	#argmin = map(soft_treshold, U)
	l1  = map(soft_treshold, X)

	#upper = X-step_least_squares*least_squares_gradient >= step_least_squares*lambdaparam
	#lower = X-step_least_squares*least_squares_gradient <= -step_least_squares*lambdaparam
	#argmin = lower*(step_least_squares*lambdaparam) - upper*(step_least_squares*lambdaparam) + np.multiply(upper+lower, X-step_least_squares*least_squares_gradient)
	#result =  (1.0/step_least_squares)*(X - argmin)  
	result =  least_squares_gradient + l1  

	return result