import numpy as np
import random
import pandas as pd
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.io
import copy
import time
from svmutil import *

def normalise(df_train,mu_train, sigma_train):
	df_train = (df_train - mu_train)/ sigma_train
	df_train.insert(0,'x0',1.0)
	return df_train


def calc_theta(df_train, target):
	transpose_matrix = df_train.transpose()
	f = np.linalg.pinv(np.dot(transpose_matrix, df_train))
	s = np.dot(transpose_matrix, target)
	theta = np.dot(f,s)
	return theta

def predict(df_train, target, theta):
	pred = [0] * len(df_train)
	for index in range(len(df_train.index)):
		pred[index] = np.dot(df_train.iloc[index], theta)
	return pred

def calc_mse(df_train, target, theta):
	pred = predict(df_train, target, theta)
	mse = 0
	for index in range(len(df_train.index)):
		mse += (pred[index] - target[index])**2 
	mse /= len(df_train)
	return mse

def linear_regression(df_train, train_target):
	theta = calc_theta(df_train, train_target)
	return (calc_mse(df_train, train_target, theta))

def calc_y(d, n):
	# np.random.seed(1)
	col = np.random.normal(0, math.sqrt(0.1), n)
	col += 2 * d['x']**2
	return col

def bias(d):
	
	target = calc_y(d, len(d.index))
	mu_d = d.mean()
	sigma_d = d.std()
	d_norm = normalise(d, mu_d, sigma_d)
	theta = calc_theta(d_norm, target)
	mse = linear_regression(d_norm, target)
	return theta, mse

def histogram(col):
	plt.hist(col, bins =10)
	plt.show()



def ridge(n):
	theta = [[] for i in xrange(7)]
	for i in xrange(100):
		d = pd.DataFrame()
		d['x'] = [random.uniform(-1, 1) for i in xrange(n)]
		d['x2'] = d['x']**2
		target = calc_y(d, len(d.index))
		mu_d = d.mean()
		sigma_d = d.std()
		d_norm = normalise(d, mu_d, sigma_d)
		for lmbda in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
			the = calc_regression_theta(d_norm, target, lmbda)
			ind = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0].index(lmbda)
			theta[ind].append(the)
	for lmbda in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
		print "For lambda",lmbda,"bias = "
		ind = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0].index(lmbda)
		calc_bias(theta[ind], 4, 500)
		print "Variance = "
		calc_variance(theta[ind], 4, 500)
			
		
	

def calc_regression_theta(df_train, target, lmbda):
	transpose_matrix = df_train.transpose()
	x = np.dot(transpose_matrix, df_train)
	iden = np.matrix(np.identity(len(df_train.columns)))
	iden[0][0] = 0
	x += lmbda * iden
	f = np.linalg.pinv(x)
	s = np.dot(transpose_matrix, target)
	theta = np.dot(f,s)
	return theta

def call_bias(n):
	mse = [[]  for i in xrange(6)]
	theta = [[]  for i in xrange(6)]
	m = 0
	# for g1 call calc_y function (no noramlise) and mse will be y (target - 1)/n
	# for g2 call df.insert. calc_y, normalise manually and call calc_theta  
	for i in xrange(100):
		d = pd.DataFrame()
		d['x'] = [1] * n
		target = calc_y(d, len(d.index))
		for j in xrange(len(d.index)):
			m += abs(target[j] - 1)
		m /= len(d.index)
		mse[0].append(m)

		#d_norm = (d - d.mean())/d.std()
		m = linear_regression(d, target) 
		mse[1].append(m)
		theta[1].append(calc_theta(d, target))
	random.seed(123)
	for i in xrange(100):
		d = pd.DataFrame()
		d['x'] = [random.uniform(-1, 1) for i in xrange(n)]
		tup = bias(d)
		mse[2].append(tup[1])
		theta[2].append(tup[0])
		d['x2'] = d['x']**2
		tup = bias(d)
		mse[3].append(tup[1])
		theta[3].append(tup[0])
		d['x3'] = d['x']**3
		tup = bias(d)
		mse[4].append(tup[1])
		theta[4].append(tup[0])
		d['x4'] = d['x']**4
		tup = bias(d)
		mse[5].append(tup[1])
		theta[5].append(tup[0])
	for i in range(0,6):
	 	histogram(mse[i])
	for i in range(0,6):
		if i == 0:
			print "For g",i+1,"bias = "
			calc_bias(np.ones(100), i+1, 500)
			print "For g",i+1,"Variance = "
			calc_variance(np.ones(100), i+1, 500)
		else:
			print "For g",i+1,"bias = "
			calc_bias(theta[i], i+1, 500)
			print "For g",i+1,"Variance = "
			calc_variance(theta[i], i+1, 500)

def calc_bias(theta, n, num):
	bias = 0
	theta = np.array(theta)
	data = [random.uniform(-1, 1) for i in xrange(num)]
	epsilon = np.random.normal(0,math.sqrt(0.1))
	for i in xrange(num):
		x = generate_data(data[i], n)
		exp_h = calc_exph(x, theta)
		e_y = calc_ey(data[i])
		p = calc_p(e_y,num,epsilon)
		bias += ((exp_h - e_y)**2) * p
	print(bias)

def calc_variance(theta, n, num):
	variance = 0
	data = [random.uniform(-1, 1) for i in xrange(num)]
	theta = np.array(theta)
	for j in xrange(100):
		thet = theta[j]
		epsilon = np.random.normal(0,math.sqrt(0.1))
		for i in xrange(num):
			x = generate_data(data[i], n)
			h = calc_h(thet,x)
			exp_h = calc_exph(x, theta)
			e_y = calc_ey(data[i])
			p = calc_p(e_y,num,epsilon)
			# print(exp_h)
			variance += ((h-exp_h)**2) * p
	print(variance/500.0)

def calc_h(x, thet):
	return np.dot(x,thet)

def generate_data(val, n):
	lst = [1]
	for i in range(1, n-1):
		lst.append(val**i)
	lst = np.array(lst)
	return lst

def calc_exph(x,theta):
	exph = 0
	for t in theta:
		exph += np.dot(t.transpose(),x)
	return exph/len(theta)

def calc_ey(x):
	val = x**2
	return 2 * val

def calc_p(ey, num, epsilon):
	num = float(num)
	val = norm.pdf(ey + epsilon,ey, math.sqrt(0.1))
	#val = np.random.normal(ey, math.sqrt(0.1))
	val *= 1.0/num
	return val

def convert(data):
	df = pd.DataFrame(data)
	col_lst = [1, 6, 7, 13, 14, 25, 28]
	for col in df.columns:
	 	if col in col_lst:
	# 		#add 3 columns
	 		df[str(col)+'neg'] = np.zeros(len(df.index))
	 		df[str(col)+'zer'] = np.zeros(len(df.index))
	 		df[str(col)+'pos'] = np.zeros(len(df.index))
	 		for row in range(len(df.index)):
				if df[col].iloc[row] == -1:
					df[str(col)+'neg'].iloc[row] = 1
				elif df[col].iloc[row] == 0:
					df[str(col)+'zer'].iloc[row] = 1
				elif df[col].iloc[row] == 1:
					df[str(col)+'pos'].iloc[row] = 1
			df.drop(col, 1, inplace=True)
	
	df.replace(-1, 0, inplace=True)
	df.columns = range(0,44)
	return df.T.to_dict().values()
	
def convert_target(arr):
	arr = list(arr)
	for i in range(len(arr)):
		if arr[i] == -1:
			arr[i] = 0
	return np.array(arr)


def svm():
	train = scipy.io.loadmat('phishing-train.mat')
	test = scipy.io.loadmat('phishing-test.mat')
	train_data = train['features']
	test_data = test['features']
	target_train = train['label']
	target_test = test['label']
	
	train_data = convert(train_data)    
	
	test_data = convert(test_data)

	prob = svm_problem(target_train.flatten(), train_data)
	
	maximum = 0.0
	maximum_c = 0
	for c in range(-6,3):
		param = svm_parameter("-c "+ str(4**c)+" -v 3 -t 0 -q")
		print "3-fold Cross-Validation Accuracy for c=" + str(c) 
		accuracy = svm_train(prob, param)
		if(accuracy > maximum):
			maximum = accuracy
			maximum_c = 4**c
	print "Best accuracy for c =" + str(maximum_c)
	param = svm_parameter("-c "+ str(maximum_c)+" -t 0 -q")
	start = time.time()
	model = svm_train(prob, param)	
	end = time.time()
	print "Time taken", end - start
	svm_predict(target_test.flatten(), test_data, model)

	maximum_d = 0
	accuracy_d = 0
	degree = [1,2,3]
	for c in range(-3, 8):
		for d in degree:
			param = svm_parameter("-c "+ str(4**c)+" -v 3 -t 1 -d "+str(d)+ " -q")
			start = time.time()
			print "for c = "+ str(c) + "and d = "+ str(d)
			accuracy = svm_train(prob, param)
			end = time.time()
			print "Time taken",end-start
			if(accuracy > maximum):
				maximum = accuracy
				maximum_c = c
				maximum_d = d
				accuracy_d = accuracy
	print "Best accuracy for c =" + str(maximum_c) +"and degree = " + str(maximum_d)

	maximum = 0.0
	maximum_c = 0
	maximum_g = 0
	accuracy_g = 0
	for c in range(-3, 8):
		for gamma in range(-7, 0):
			param = svm_parameter("-c "+ str(4**c)+" -v 3 -t 2 -g "+str(4**gamma)+ " -q")
			start =  time.time()
			print "for c = "+ str(c) + "and g = "+ str(gamma)
			accuracy = svm_train(prob, param)
			end = time.time()
			
			print "Time taken",end - start
			if(accuracy > maximum):
				maximum = accuracy
				maximum_c = c
				maximum_g = gamma
				accuracy_g = accuracy
	print "Best accuracy for c =" + str(maximum_c) +"and gamma = " + str(maximum_g)
	# if accuracy_g > accuracy_d:
	# 	param = svm_parameter("-c "+ str(maximum_c)+" -t 2 -g "+str(maximum_g)+ " -q")
	# elif accuracy_d > accuracy_g:
	# 	param = svm_parameter("-c "+ str(maximum_c)+" -t 1 -d "+str(maximum_d)+ " -q")
	# model = svm_train(prob, param)	

	# svm_predict(target_test.flatten(), test_data, model)


if __name__ == '__main__':

	print "Linear Regression for 10 Samples in dataset" 
	call_bias(10)	
	print "Linear Regression For 100 Samples in dataset"
	call_bias(100)
	print "Ridge Regression For 100 Samples in dataset" 
	ridge(100)
	print "Linear and Kernel SVM"
	svm()
	
