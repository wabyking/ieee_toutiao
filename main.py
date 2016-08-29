#coding=utf-8
import os
import random,math
import cPickle as pickle
import pandas as pd
import ConfigParser
from sklearn.linear_model import LinearRegression 
import ndcg
import numpy as np
conf = ConfigParser.ConfigParser()   
conf.read("arg.cfg") 


def loadData():

	user=pd.read_csv(conf.get("path","user_path"),header=None,sep="\t",names=["uid","labels","word_seq","char_seq"])
	user["labels"]=user["labels"].str.split("/")
	user["word_seq"]=user["word_seq"].str.split("/")
	user["char_seq"]=user["char_seq"].str.split("/")
	
	question=pd.read_csv(conf.get("path","question_path"),header=None,sep="\t",names=["qid","_label","word_seq","char_seq","likesLen","answersLen","goodAnswerLen"])
	question["word_seq"]=question["word_seq"].str.split("/")
	question["char_seq"]=question["char_seq"].str.split("/")
	

	train=pd.read_csv(conf.get("path","train"),names=["qid","uid","label"],sep="\t")
	

	validate=pd.read_csv(conf.get("path","validate"))
	return user,question,train,validate

def user_anwsers(group):

	return len(group[group["label"]==1])


def word_overlap(row):
	if type(row["word_seq_question"] )==float or  type(row["word_seq_user"]) ==float :
		return 0

	return sum( [ word in row["word_seq_user"]  for word in row["word_seq_question"] ])
	
def char_overlap(row):
	if type(row["char_seq_question"] )==float or  type(row["char_seq_user"]) ==float :
		return 0
	return sum( [ word in row["char_seq_user"]  for word in row["char_seq_question"] ])

def fill(num):
	if math.isnan(num):
		return 0
	else:
		return num
def getFeatures(df,fresh=False):
	names=["user_anwsers","word_overlap","char_overlap"]
	# user_features["user_anwsers"]=
	user_feature_pkl=conf.get("temp","user_feature")
	if fresh==False and os.path.exists(user_feature_pkl):
		print "the feature of user have been processed"
		user_features=pickle.load(open(user_feature_pkl,'r'))
	else:
		temp=df.groupby("uid").apply(user_anwsers)
		user_features=pd.DataFrame({"uid":temp.index,"user_anwsers":temp}) 
		pickle.dump(user_features,open(user_feature_pkl,"w"))	
	
	features=df.merge(user_features,how="left",)
	
	features["user_anwsers"]=features["user_anwsers"].apply(fill)
	# print features
	# print features.columns
	# exit()
	
	features["word_overlap"]=features.apply(word_overlap,axis=1)
	features["char_overlap"]=features.apply(char_overlap,axis=1)

	return features,names
	


def dataSplit(df,rate=0.8):

	size=len(df)
	questions= df["qid"].unique()

	flags=[True] * int(size*rate) + [False] *  (size-int(size*rate))
	random.seed(822)
	random.shuffle(flags)
	
	trainQuestions= [questions[i] for i in range(len(questions)) if flags[i]==True]
	# reverse_flags=[not item  for item in flags]
	# testQustions= df["question"][reverse_flags]
	train=df[df.qid.isin(trainQuestions)]
	test=df[~df.qid.isin(trainQuestions)]
	
	return train,test
def ndcg_apply(group):
	seq= group.sort_values(by="predict")["label"]
	return ndcg.evaluation(seq)

def evaluation(predict,groundTruth):
	groundTruth["predict"]=predict
	return groundTruth.groupby("qid").apply(ndcg_apply).mean()

def write2file(df):
	df[["qid","uid","label"]].to_csv("temp.csv",index=False)

def main(option="online"):
	user,question,train,validate=loadData()

	temp=pd.merge(train,question,how="inner",on="qid")
	train= temp.merge(user,how="inner",on="uid",suffixes=('_question', '_user'))
	
	temp=pd.merge(validate,question,how="inner")
	validate= temp.merge(user,how="inner",on="uid",suffixes=('_question', '_user'))

	train_features,names=getFeatures(train)
	print "train features over"
	

	if option=="online":

		validate_features,names=getFeatures(validate)
		print "validate features over"
		clf = LinearRegression()
		clf.fit(train_features[names], train["label"])
		
		print train_features[names]
		predicted=clf.predict(validate_features[names])
		print clf.coef_
		validate_features["label"]=predicted
		validate_features["label"]=validate_features["label"]/validate_features["label"].max()#+ np.random.standard_normal(len(validate_features))/10
		write2file(validate_features)
	elif option=="offline":
		
		train,test=dataSplit(train_features)
		clf = LinearRegression()
		clf.fit(train[names], train["label"])
		print clf.coef_
		predicted=clf.predict(test[names])

		print evaluation(predicted,test[["qid","label"]])
	else:
		validate_features,names=getFeatures(validate)
		label=validate_features["user_anwsers"].fillna(0)
		validate_features["label"]=label/label.max()
	
		write2file(validate_features)
	




if __name__=="__main__":
	main()