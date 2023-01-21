import sys
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model,load_model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

input_training_dir = "Input/Training/AD/"
input_training_data_dir = input_training_dir + "Data/"
input_training_tst_data_dir = input_training_data_dir + "TST/"
input_training_an_data_dir = input_training_data_dir + "AN/"
input_training_model_dir = input_training_dir + "Model/"
input_training_AE_dir = input_training_model_dir + "AE/"
input_training_HC_dir = input_training_model_dir + "HC/"
input_training_SC_dir = input_training_model_dir + "SC/"

output_training_dir = "Output/Training/AD/"
output_training_classifications_dir = output_training_dir + "Classifications/"
input_inference_AD = "Input/Inference/AD/"
input_inference_AD_Data = input_inference_AD + "Data/"
input_inference_AD_Model = input_inference_AD + "Model/"

input_inference_model_dir = input_inference_AD + "Model/"
input_inference_AE_dir = input_inference_model_dir + "AE/"
input_inference_HC_dir = input_inference_model_dir + "HC/"
input_inference_SC_dir = input_inference_model_dir + "SC/"

def load_data(mode):
	data = None

	if mode == "training":
		data = {}
		data["N"] = {}
		data["A"] = {}
		for filename in os.listdir(input_training_tst_data_dir):
			cnn = filename.split(".")[0].split("_")[0]
			monitor_layer = filename.split(".")[0].split("_")[1]
			try:
				data["N"][cnn][monitor_layer] = pd.read_csv(input_training_tst_data_dir + filename)
			except KeyError:
				data["N"][cnn] = {}
				data["N"][cnn][monitor_layer] = pd.read_csv(input_training_tst_data_dir + filename)
		for filename in os.listdir(input_training_an_data_dir):
			cnn = filename.split(".")[0].split("_")[0]
			monitor_layer = filename.split(".")[0].split("_")[1]
			try:
				data["A"][cnn][monitor_layer] = pd.read_csv(input_training_an_data_dir + filename)
			except KeyError:
				data["A"][cnn] = {}
				data["A"][cnn][monitor_layer] = pd.read_csv(input_training_an_data_dir + filename)
	elif mode == "inference":
			data = {}
			for filename in os.listdir(input_inference_AD_Data):
				cnn = filename.split(".")[0].split("_")[0]
				monitor_layer = filename.split(".")[0].split("_")[1]
				try:
					data[cnn][monitor_layer] = pd.read_csv(input_inference_AD_Data + filename)
				except KeyError:
					data[cnn] = {}
					data[cnn][monitor_layer] = pd.read_csv(input_inference_AD_Data + filename)
     
        

	return data

def read_centroids(cnn, monitor_layer):
	centroids = {}

	file = open(input_training_HC_dir + "centroids_" + str(cnn) + "_" + str(monitor_layer) + ".txt", "r")
	lines = file.readlines()
	for line in lines:
		line = line.replace("\n","")
		line = line.replace(" ","")
		line = line.replace("[","")
		line = line.replace("]","")
		tokens = line.split(":")
		centroid_coordinates = tokens[-1].split(",")
		centroids[int(tokens[0])] = []
		for centroid_coordinate in centroid_coordinates:
			centroids[int(tokens[0])].append(float(centroid_coordinate))
	file.close()

	return centroids

def read_threshold(cnn, monitor_layer):
	threshold = -1
	if ad_technique == "AE":
		file = open(input_training_AE_dir + "mse_" + cnn + "_" + monitor_layer + ".txt", "r")
		threshold = float(file.readline())
		file.close()
	elif ad_technique == "HC":
		file = open(input_training_HC_dir + "threshold_" + cnn + "_" + monitor_layer + ".txt", "r")
		threshold = float(file.readline())
		file.close()
	elif ad_technique == "SC":
		file = open(input_training_SC_dir + "threshold_" + cnn + "_" + monitor_layer + ".txt", "r")
		threshold = float(file.readline())
		file.close()
	return threshold

def read_distribution(cnn, monitor_layer):

	distribution = {}
	file = open(input_training_SC_dir +  "distribution_" + str(cnn) + "_" + str(monitor_layer) + ".txt", "r")
	lines = file.readlines()
	distribution["weights"] = []
	distribution["means"] = []
	distribution["covariances"] = []

	n_means_lines = [0,0]
	n_covariances_lines = [0, 0]
	for idx,line in enumerate(lines):
		if line.split(":")[0] == "means":
			n_means_lines[0] = idx
		if line.split(":")[0] == "covariances":
			n_covariances_lines[0] = idx
	n_means_lines[1] = n_covariances_lines[0]-1
	n_covariances_lines[1] = len(lines)

	temp = []
	vec_values = lines[0].split(":")[1].replace("\n","").split(",")
	for vec_value in vec_values:
		temp.append(float(vec_value))
	distribution["weights"] = temp

	for idx,line in enumerate(lines[n_means_lines[0]+1:n_means_lines[1]+1]):
		temp = []
		vec_values = line.split(":")[1].replace("\n","").split(",")
		for vec_value in vec_values:
			temp.append(float(vec_value))
		distribution["means"].append(temp)
	matrices_idx = 0
	for idx,line in enumerate(lines[n_covariances_lines[0]+1:n_covariances_lines[1]+1]):
		if line.find("matrix") != -1:
			distribution["covariances"].append([])
			matrices_idx = matrices_idx+1
		else:
			temp = []
			vec_values = line.split(":")[1].replace("\n","").split(",")
			for vec_value in vec_values:
				temp.append(float(vec_value))
			distribution["covariances"][matrices_idx-1].append(temp)
	distribution["weights"] = np.array(distribution["weights"])
	distribution["means"] = np.array(distribution["means"])
	distribution["covariances"] = np.array(distribution["covariances"])
	file.close()

	return distribution

def load_models(mode, ad_technique):

	models = None
	thresholds = None

	if mode == "training":
		models = {}
		thresholds = {}
		if ad_technique == "AE":
			for filename in os.listdir(input_training_AE_dir):
				if filename.endswith(".h5"):
					cnn = filename.split(".")[0].split("_")[0]
					monitor_layer = filename.split(".")[0].split("_")[1]
					try:
						models[cnn][monitor_layer] = load_model(input_training_AE_dir + filename)
						thresholds[cnn][monitor_layer] = read_threshold(cnn, monitor_layer)
					except:
						models[cnn] = {}
						thresholds[cnn] = {}
						models[cnn][monitor_layer] = load_model(input_training_AE_dir + filename)
						thresholds[cnn][monitor_layer] = read_threshold(cnn, monitor_layer)
		elif ad_technique == "HC":
			for filename in os.listdir(input_training_HC_dir):
				if filename.split("_")[0] == "centroids":
					cnn = filename.split(".")[0].split("_")[1]
					monitor_layer = filename.split(".")[0].split("_")[2]
					try:
						models[cnn][monitor_layer] = read_centroids(cnn, monitor_layer)
						thresholds[cnn][monitor_layer] = read_threshold(cnn, monitor_layer)
					except:
						models[cnn] = {}
						thresholds[cnn] = {}
						models[cnn][monitor_layer] = read_centroids(cnn, monitor_layer)
						thresholds[cnn][monitor_layer] = read_threshold(cnn, monitor_layer)
		elif ad_technique == "SC":
			for filename in os.listdir(input_training_SC_dir):
				if filename.split("_")[0] == "distribution":
					cnn = filename.split(".")[0].split("_")[1]
					monitor_layer = filename.split(".")[0].split("_")[2]
					try:
						models[cnn][monitor_layer] = read_distribution(cnn, monitor_layer)
						thresholds[cnn][monitor_layer] = read_threshold(cnn, monitor_layer)
					except:
						models[cnn] = {}
						thresholds[cnn] = {}
						models[cnn][monitor_layer] = read_distribution(cnn, monitor_layer)
						thresholds[cnn][monitor_layer] = read_threshold(cnn, monitor_layer)
	elif mode == "inference":
		models = {}
		thresholds = {}
		if ad_technique == "AE":
			for filename in os.listdir(input_inference_AE_dir):
				if filename.endswith(".h5"):
					cnn = filename.split(".")[0].split("_")[0]
					monitor_layer = filename.split(".")[0].split("_")[1]
					try:
						models[cnn][monitor_layer] = load_model(input_inference_AE_dir + filename)
						thresholds[cnn][monitor_layer] = read_threshold(cnn, monitor_layer)
					except:
						models[cnn] = {}
						thresholds[cnn] = {}
						models[cnn][monitor_layer] = load_model(input_inference_AE_dir + filename)
						thresholds[cnn][monitor_layer] = read_threshold(cnn, monitor_layer)
		elif ad_technique == "HC":
			for filename in os.listdir(input_inference_HC_dir):
				if filename.split("_")[0] == "centroids":
					cnn = filename.split(".")[0].split("_")[1]
					monitor_layer = filename.split(".")[0].split("_")[2]
					try:
						models[cnn][monitor_layer] = read_centroids(cnn, monitor_layer)
						thresholds[cnn][monitor_layer] = read_threshold(cnn, monitor_layer)
					except:
						models[cnn] = {}
						thresholds[cnn] = {}
						models[cnn][monitor_layer] = read_centroids(cnn, monitor_layer)
						thresholds[cnn][monitor_layer] = read_threshold(cnn, monitor_layer)
		elif ad_technique == "SC":
			for filename in os.listdir(input_inference_SC_dir):
				if filename.split("_")[0] == "distribution":
					cnn = filename.split(".")[0].split("_")[1]
					monitor_layer = filename.split(".")[0].split("_")[2]
					try:
						models[cnn][monitor_layer] = read_distribution(cnn, monitor_layer)
						thresholds[cnn][monitor_layer] = read_threshold(cnn, monitor_layer)
					except:
						models[cnn] = {}
						thresholds[cnn] = {}
						models[cnn][monitor_layer] = read_distribution(cnn, monitor_layer)
						thresholds[cnn][monitor_layer] = read_threshold(cnn, monitor_layer)
	return models, thresholds

def soft_cluster_dataset(dataset, reuse_parameters, clustering_parameters_in):
	clustered_dataset = dataset.copy()
	clustering_parameters = {}

	if reuse_parameters == 0:
		gaussian_mixture = GaussianMixture(n_components=n_labeling_clusters, covariance_type='full', random_state=0).fit(dataset)
		clustering_parameters["weights"] = gaussian_mixture.weights_
		clustering_parameters["means"] = gaussian_mixture.means_
		clustering_parameters["covariances"] = gaussian_mixture.covariances_
		labels_probability = gaussian_mixture.predict_proba(dataset)
		scores = gaussian_mixture.score_samples(dataset)
		for j in range(n_labeling_clusters):
			temp = []
			for i in range(0,len(dataset)):
				temp.append(labels_probability[i][j])
			clustered_dataset["Cluster_" + str(j)] = temp
		clustered_dataset["Scores"] = scores
		
	elif reuse_parameters == 1:
		n_labeling_clusters = len(clustering_parameters_in["weights"])
		gaussian_mixture = GaussianMixture(n_components=n_labeling_clusters, covariance_type='full')
		gaussian_mixture.weights_ = clustering_parameters_in["weights"]
		gaussian_mixture.means_ = clustering_parameters_in["means"]
		gaussian_mixture.covariances_ = clustering_parameters_in["covariances"]
		gaussian_mixture.precisions_cholesky_ = _compute_precision_cholesky(clustering_parameters_in["covariances"], 'full')
		labels_probability = gaussian_mixture.predict_proba(dataset)
		scores = gaussian_mixture.score_samples(dataset)
		for j in range(n_labeling_clusters):
			temp = []
			for i in range(0,len(dataset)):
				temp.append(labels_probability[i][j])
			clustered_dataset["Cluster_" + str(j)] = temp
		clustered_dataset["Scores"] = scores


	return clustered_dataset, clustering_parameters

def classify(data, models, thresholds, ad_technique):
	
	classifications = []

	if ad_technique == "AE":
		monitor_wise_classifications = {}
		for monitor_layer in data:
			monitor_wise_classifications[monitor_layer] = []
			data_np_array = np.array(data[monitor_layer])
			recon = models[monitor_layer].predict(data_np_array, verbose=0)
			for idx,elem in enumerate(data_np_array):
				error = mean_squared_error(data_np_array[idx],recon[idx])
				if error > thresholds[monitor_layer]:
					monitor_wise_classifications[monitor_layer].append("A")
				else:
					monitor_wise_classifications[monitor_layer].append("N")
			

		monitor_layers = list(monitor_wise_classifications.keys())

		for idx,elem in enumerate(monitor_wise_classifications[monitor_layers[0]]):
			temp = []
			for monitor_layer in monitor_layers:
				temp.append(monitor_wise_classifications[monitor_layer][idx])
			if temp.count("A") > 0:
				classifications.append("A")
			else:
				classifications.append("N")
		
	elif ad_technique == "HC":
		monitor_wise_classifications = {}
		for monitor_layer in data:
			monitor_wise_classifications[monitor_layer] = []
			data_np_array = np.array(data[monitor_layer])
			for sample in data_np_array:
				min_local_distance = float("inf")
				for centroid in models[monitor_layer]:
					dist = np.linalg.norm(sample-np.array(models[monitor_layer][centroid]))
					if dist < min_local_distance:
						min_local_distance = dist
				if dist > thresholds[monitor_layer]:
					monitor_wise_classifications[monitor_layer].append("A")
				else:
					monitor_wise_classifications[monitor_layer].append("N")

		monitor_layers = list(monitor_wise_classifications.keys())

		for idx,elem in enumerate(monitor_wise_classifications[monitor_layers[0]]):
			temp = []
			for monitor_layer in monitor_layers:
				temp.append(monitor_wise_classifications[monitor_layer][idx])
			if temp.count("A") > 0:
				classifications.append("A")
			else:
				classifications.append("N")
		
	elif ad_technique == "SC":
		monitor_wise_classifications = {}
		for monitor_layer in data:
			monitor_wise_classifications[monitor_layer] = []
			clustered_data, ignore = soft_cluster_dataset(data[monitor_layer], 1, models[monitor_layer])
			for index, row in clustered_data.iterrows():
				if row["Scores"] < thresholds[monitor_layer]:
					monitor_wise_classifications[monitor_layer].append("A")
				else:
					monitor_wise_classifications[monitor_layer].append("N")


		monitor_layers = list(monitor_wise_classifications.keys())

		for idx,elem in enumerate(monitor_wise_classifications[monitor_layers[0]]):
			temp = []
			for monitor_layer in monitor_layers:
				temp.append(monitor_wise_classifications[monitor_layer][idx])
			if temp.count("A") > 0:
				classifications.append("A")
			else:
				classifications.append("N")
	return classifications

def vote(d_20_classifications, d_32_classifications, d_56_classifications):
	classifications = []

	for idx,classification in enumerate(d_20_classifications):
		temp = [d_20_classifications[idx], d_32_classifications[idx], d_56_classifications[idx]]
		if temp.count("A") > 1:
			classifications.append("A")
		else:
			classifications.append("N")
	
	return classifications

def get_performance_metrics(n_classifications, a_classifications):

	tp = 0
	tn = 0
	fp = 0
	fn = 0

	for classification in n_classifications:
		if classification == "N":
			tn = tn+1
		else:
			fp = fp+1
	
	for classification in a_classifications:
		if classification == "A":
			tp = tp+1
		else:
			fn = fn+1

	try:
		accuracy = (tp+tn)/(tp+tn+fp+fn)
	except ZeroDivisionError:
		print("Accuracy could not be computed because the denominator was 0")
		accuracy = "undefined"

	try:
		precision = tp/(tp+fp)
	except ZeroDivisionError:
		print("Precision could not be computed because the denominator was 0")
		precision = "undefined"

	try:
		recall = tp/(tp+fn)
	except ZeroDivisionError:
		print("Recall could not be computed because the denominator was 0")
		recall = "undefined"
		
	try:
		f1 = 2*tp/(2*tp+fp+fn)
	except ZeroDivisionError:
		print("F1 could not be computed because the denominator was 0")
		f1 = "undefined"	
	

	return accuracy, precision, recall, f1, tp, tn, fp, fn

def write_performance_metrics(accuracy, precision, recall, f1, tp, tn, fp, fn, cnn):

	classifications_file = open(output_training_classifications_dir + cnn + "_classifications.txt","w")
	classifications_file.write("TP="+str(tp)+"\nTN="+str(tn)+"\nFP="+str(fp)+"\nFN="+str(fn)+"\nAccuracy="+str(accuracy)+"\nPrecision="+str(precision)+"\nRecall="+str(recall)+"\nF1="+str(f1))
	classifications_file.close()

	return None

try:
	debug_mode = int(sys.argv[1])
	mode = sys.argv[2]
	ad_technique = sys.argv[3]
except:
	print("Not enough input arguments. Please, insert the input arguments.")
	



if mode == "training":
	data = load_data(mode)
	models, thresholds = load_models(mode, ad_technique)

	d_20_classifications = {}
	d_32_classifications = {}
	d_56_classifications = {}
	d_classifications = {}
	d_20_classifications["N"] = classify(data["N"]["20"], models["20"], thresholds["20"], ad_technique)
	d_20_classifications["A"] = classify(data["A"]["20"], models["20"], thresholds["20"], ad_technique)
	if debug_mode == 1:
		accuracy, precision, recall, f1, tp, tn, fp, fn = get_performance_metrics(d_20_classifications["N"], d_20_classifications["A"])
		print("Classification results (ResNet20 only):")
		print("TP="+str(tp))
		print("TN="+str(tn))
		print("FP="+str(fp))
		print("FN="+str(fn))
		print("Accuracy: " + str(accuracy))
		print("Precision: " + str(precision))
		print("Recall: " + str(recall))
		print("F1: " + str(f1))
		write_performance_metrics(accuracy, precision, recall, f1, tp, tn, fp, fn, "ResNet20")
	d_32_classifications["N"] = classify(data["N"]["32"], models["32"], thresholds["32"], ad_technique)
	d_32_classifications["A"] = classify(data["A"]["32"], models["32"], thresholds["32"], ad_technique)
	if debug_mode == 1:
		accuracy, precision, recall, f1, tp, tn, fp, fn = get_performance_metrics(d_32_classifications["N"], d_32_classifications["A"])
		print("Classification results (ResNet32 only):")
		print("TP="+str(tp))
		print("TN="+str(tn))
		print("FP="+str(fp))
		print("FN="+str(fn))
		print("Accuracy: " + str(accuracy))
		print("Precision: " + str(precision))
		print("Recall: " + str(recall))
		print("F1: " + str(f1))
		write_performance_metrics(accuracy, precision, recall, f1, tp, tn, fp, fn, "ResNet32")
	d_56_classifications["N"] = classify(data["N"]["56"], models["56"], thresholds["56"], ad_technique)
	d_56_classifications["A"] = classify(data["A"]["56"], models["56"], thresholds["56"], ad_technique)
	if debug_mode == 1:
		accuracy, precision, recall, f1, tp, tn, fp, fn = get_performance_metrics(d_56_classifications["N"], d_56_classifications["A"])
		print("Classification results (ResNet56 only):")
		print("TP="+str(tp))
		print("TN="+str(tn))
		print("FP="+str(fp))
		print("FN="+str(fn))
		print("Accuracy: " + str(accuracy))
		print("Precision: " + str(precision))
		print("Recall: " + str(recall))
		print("F1: " + str(f1))
		write_performance_metrics(accuracy, precision, recall, f1, tp, tn, fp, fn, "ResNet56")
	d_classifications["N"] = vote(d_20_classifications["N"], d_32_classifications["N"], d_56_classifications["N"])
	d_classifications["A"] = vote(d_20_classifications["A"], d_32_classifications["A"], d_56_classifications["A"])
	if debug_mode == 1:
		accuracy, precision, recall, f1, tp, tn, fp, fn = get_performance_metrics(d_classifications["N"], d_classifications["A"])
		print("Classification results (ResNet20/32/56):")
		print("TP="+str(tp))
		print("TN="+str(tn))
		print("FP="+str(fp))
		print("FN="+str(fn))
		print("Accuracy: " + str(accuracy))
		print("Precision: " + str(precision))
		print("Recall: " + str(recall))
		print("F1: " + str(f1))
		write_performance_metrics(accuracy, precision, recall, f1, tp, tn, fp, fn, "ResNet20-32-56")
	

elif mode == "inference":
	data = load_data(mode)
	models, thresholds = load_models(mode, ad_technique)
	d_20_classifications = {}
	d_32_classifications = {}
	d_56_classifications = {}
	d_classifications = {}
	d_20_classifications = classify(data["20"], models["20"], thresholds["20"], ad_technique)
	d_32_classifications= classify(data["32"], models["32"], thresholds["32"], ad_technique)
	d_56_classifications = classify(data["56"], models["56"], thresholds["56"], ad_technique)
	d_classifications= vote(d_20_classifications, d_32_classifications, d_56_classifications)
	print(d_classifications)
	












