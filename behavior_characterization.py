import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import numpy as np

from tensorflow.keras.layers import Dense,Input,Concatenate
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from scipy.spatial import distance

from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

input_training_dir = "Input/Training/BC/"
input_training_data_dir = input_training_dir + "Data/"

output_training_dir = "Output/Training/BC/"
output_training_model_dir = output_training_dir + "Model/"
output_training_AE_dir = output_training_model_dir + "AE/"
output_training_HC_dir = output_training_model_dir + "HC/"
output_training_SC_dir = output_training_model_dir + "SC/"

debug_mode = -1
mode = ""
ad_technique = ""



def cluster_dataset(dataset, reuse_parameters, clustering_parameters_in):
	clustered_dataset = dataset.copy()
	clustering_parameters = {}

	
	if reuse_parameters == 0:
		if clustering_technique == "agglomerative":
			cluster_configuration = AgglomerativeClustering(n_clusters=n_clusters, affinity='cityblock', linkage='average')
			cluster_labels = cluster_configuration.fit_predict(clustered_dataset)
		elif clustering_technique == "kmeans":
			kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(clustered_dataset)
			cluster_labels = kmeans.labels_

		clustered_dataset["Cluster"] = cluster_labels
		cluster_labels = cluster_labels.tolist()
		used = set();
		clusters = [x for x in cluster_labels if x not in used and (used.add(x) or True)]

		instances_sets = {}
		centroids = {}
		
		for cluster in clusters:
			instances_sets[cluster] = []
			centroids[cluster] = []
		
		temp = clustered_dataset
		for index, row in temp.iterrows():
			instances_sets[int(row["Cluster"])].append(row.values.tolist())
		
		n_features_per_instance = len(instances_sets[0][0])-1
		
		for instances_set_label in instances_sets:
			instances = instances_sets[instances_set_label]
			for idx, instance in enumerate(instances):
				instances[idx] = instance[0:n_features_per_instance]
			for i in range(0,n_features_per_instance):
				values = []
				for instance in instances:
					values.append(instance[i])
				centroids[instances_set_label].append(np.mean(values))
				
		clustering_parameters = centroids
			
	elif reuse_parameters == 1:
		clusters = []
		for index, instance in clustered_dataset.iterrows():
			min_value = float('inf')
			min_centroid = -1
			for centroid in clustering_parameters_in:
				centroid_coordinates = np.array([float(i) for i in clustering_parameters_in[centroid]])
				dist = np.linalg.norm(instance.values-centroid_coordinates)
				if dist<min_value:
					min_value = dist
					min_centroid = centroid
			clusters.append(min_centroid)
		
		clustered_dataset["Cluster"] = clusters
		

	return clustered_dataset, clustering_parameters

def load_data(mode):
	data = {}
	for filename in os.listdir(input_training_data_dir):
		if filename != ".DS_Store":
			data[filename.split(".")[0]] = pd.read_csv(input_training_data_dir + filename)
	return data

def autoencoder(hidden_neurons,latent_code_dimension):
	input_layer = Input(shape=(512,)) # Input
	encoder = Dense(hidden_neurons,activation="relu")(input_layer) # Encoder
	code = Dense(latent_code_dimension)(encoder) # Code
	decoder = Dense(hidden_neurons,activation="relu")(code) # Decoder
	output_layer = Dense(512,activation="linear")(decoder) # Output
	model = Model(inputs=[input_layer],outputs=[output_layer])
	model.compile(optimizer="adam",loss="mse")
	if debug_mode == 1:
		model.summary()
	return model

def train_autoencoder(normal_data, validation_split_percentage, hidden_neurons, latent_code_dimension, epochs, type):

	train_data = np.array(normal_data)
	assert latent_code_dimension < 512, print("Il codice dell'autoencoder deve essere strettamente minore del numero di features")
	model = autoencoder(hidden_neurons,latent_code_dimension)
	if debug_mode == 1:
		history = model.fit(train_data,train_data,epochs=epochs,shuffle=True,verbose=0,validation_split=validation_split_percentage,callbacks= [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'), ModelCheckpoint(output_training_AE_dir + type + ".h5",monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
	else:
		history = model.fit(train_data,train_data,epochs=epochs,shuffle=True,verbose=0,validation_split=validation_split_percentage,callbacks= [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'), ModelCheckpoint(output_training_AE_dir + type + ".h5",monitor='val_loss', save_best_only=True, mode='min', verbose=0)])

	return model

def soft_cluster_dataset(dataset, reuse_parameters, clustering_parameters_in):
	clustered_dataset = dataset.copy()
	clustering_parameters = {}

	if reuse_parameters == 0:
		if soft_clustering_technique == "gmm":
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
		if soft_clustering_technique == "gmm":
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

def compute_hc_threshold(centroids, validation_data):
	threshold = -1
	validation_data = np.array(validation_data)
	min_distances = []
	for sample in validation_data:
		min_local_distance = float("inf")
		for centroid in centroids:
			
			dist = np.linalg.norm(sample-np.array(centroids[centroid]))
			if dist < min_local_distance:
				min_local_distance = dist
		min_distances.append(min_local_distance)

	threshold = max(min_distances)

	return threshold

def compute_sc_threshold(distribution, validation_data):
	threshold = -1
	validation_data, ignore = soft_cluster_dataset(validation_data, 1, distribution)
	threshold = min(validation_data["Scores"])


	return threshold

def save_mse(mse, type):
	file = open(output_training_AE_dir + "mse_" + type + ".txt", "w")
	file.write(str(mse))
	file.close()
	return None

def save_hc_model(centroids, threshold, type):

	file = open(output_training_HC_dir + "centroids_" + type + ".txt", "w")
	for idx, centroid in enumerate(centroids):
		if idx<len(centroids)-1:
			file.write(str(centroid) + ":" + str(centroids[centroid]) + "\n")
		else:
			file.write(str(centroid) + ":" + str(centroids[centroid]))
	file.close()

	file = open(output_training_HC_dir + "threshold_" + type + ".txt", "w")
	file.write(str(threshold))
	file.close()

	return None

def save_sc_model(distribution, threshold, type):
	file = open(output_training_SC_dir + "distribution_" + type + ".txt", "w")
	for idx, clustering_parameter in enumerate(distribution):
		if clustering_parameter == "weights":
			file.write(str(clustering_parameter) + ":")
			for w_idx, weight in enumerate(distribution[clustering_parameter]):
				if w_idx<len(distribution[clustering_parameter])-1:
					file.write(str(round(weight,4)) + ",")
				else:
					file.write(str(round(weight,4)) + "\n")
		elif clustering_parameter == "means":
			file.write(str(clustering_parameter) + ":\n")
			for m_idx,means_vector in enumerate(distribution[clustering_parameter]):
				file.write("\tvector " + str(m_idx) + ":")
				for elem_idx,mean_elem in enumerate(means_vector):
					if elem_idx<len(means_vector)-1:
						file.write(str(round(mean_elem,4)) + ",")
					else:
						file.write(str(round(mean_elem,4)) + "\n")
		elif clustering_parameter == "covariances":
			file.write(str(clustering_parameter) + ":\n")
			for cov_idx,covariances_matrix in enumerate(distribution[clustering_parameter]):
				file.write("\tmatrix " + str(cov_idx) + ":\n")
				for mat_vec_idx,mat_vec in enumerate(covariances_matrix):
					file.write("\t\tvector " + str(mat_vec_idx) +":")
					for elem_idx,mat_vec_elem in enumerate(mat_vec):
						if elem_idx<len(mat_vec)-1:
							file.write(str(round(mat_vec_elem,12)) + ",")
						else:
							file.write(str(round(mat_vec_elem,12)) + "\n")
	file.close()
	
	file = open(output_training_SC_dir + "threshold_" + type + ".txt", "w")
	file.write(str(threshold))
	file.close()

try:
	debug_mode = int(sys.argv[1])
	ad_technique = sys.argv[2]
	validation_percentage = float(sys.argv[3])
except IndexError:
	print("Not enough input arguments. Please, insert the input arguments.")
	sys.exit()


data = load_data(mode)

if ad_technique == "AE":
	hidden_neurons = 30
	latent_code_dimension = 30
	epochs = 9999
	model = {}
	for type in data:
		model[type] = train_autoencoder(data[type], validation_percentage, hidden_neurons, latent_code_dimension, epochs, type)
		train_recons = model[type].predict(data[type])
		train_mse = mean_squared_error(data[type],train_recons)
		save_mse(train_mse, type)

elif ad_technique == "HC":
	n_clusters = 2
	clustering_technique = "kmeans"
	model = {}
	for type in data:
		train_data, validation_data = train_test_split(data[type],test_size=validation_percentage,shuffle=True)
		ignore, model[type] = cluster_dataset(train_data, 0, None)
		threshold = compute_hc_threshold(model[type], validation_data)
		save_hc_model(model[type],threshold,type)

elif ad_technique == "SC":
	n_labeling_clusters = 10
	soft_clustering_technique = "gmm"
	model = {}
	for type in data:
		train_data, validation_data = train_test_split(data[type],test_size=validation_percentage,shuffle=True)
		ignore, model[type] = soft_cluster_dataset(train_data, 0, None)
		threshold = compute_sc_threshold(model[type], validation_data)
		save_sc_model(model[type],threshold,type)





















	
