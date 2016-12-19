import scipy.io
from svmutil import * 
import numpy as np
import pandas as pd
def rbf():

	train = scipy.io.loadmat('phishing-train.mat')
	test = scipy.io.loadmat('phishing-test.mat')
	train_data = train['features']
	test_data = test['features']
	target_train = train['label']
	target_test = test['label']
	
	train_data = convert(train_data)    
	
	test_data = convert(test_data)
	prob = svm_problem(target_train.flatten(), train_data)
	param = svm_parameter("-c "+ str(7)+" -t 2 -g "+str(4**(-1))+ " -q")
	model = svm_train(prob, param)	
	svm_predict(target_test.flatten(), test_data, model)
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


if __name__ == '__main__':
	rbf()

