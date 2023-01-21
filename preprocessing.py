import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

np.seterr(divide='ignore', invalid='ignore')


input_training_dir = "Input/Training/PP/"
output_training_dir = "Output/Training/PP/"
input_training_data_dir = input_training_dir + "Data/"
output_training_data_dir = output_training_dir + "Data/"
output_train_data_dir = output_training_data_dir + "TR/"
output_test_data_dir = output_training_data_dir + "TST/"
output_anomalous_data_dir = output_training_data_dir + "AN/"
output_training_preprocessing_dir = output_training_dir + "Preprocessing/"
output_channels_normalization_dir = output_training_preprocessing_dir + "Channels_Normalization/"
output_channels_compression_dir = output_training_preprocessing_dir + "Channels_Compression/"
output_final_compression_dir = output_training_preprocessing_dir + "Final_Compression/"

input_inference_dir = "Input/Inference/PP/"
input_inference_data_dir = input_inference_dir + "Data/"
output_inference_dir = "Output/Inference/PP/"
output_inference_data_dir= output_inference_dir + "Data/"

mode = ""
a_type = ""
n_monitors = -1
n_components = -1
n_components_final = -1
normalization_technique = "min-max"

def load_data(mode, a_type, n_monitors):

	if mode == "training":
		data = {}
		data["N"] = {}
		data["N"]["20"] = {}
		data["N"]["32"] = {}
		data["N"]["56"] = {}
		data["A"] = {}
		data["A"]["20"] = {}
		data["A"]["32"] = {}
		data["A"]["56"] = {}
		for i in range(1,n_monitors+1):
			data["N"]["20"]["M"+str(i)] = np.load(input_training_data_dir + "N_Cifar10_"+str(n_monitors)+"M/" + "20_M" + str(i) + ".npy")
			data["N"]["32"]["M"+str(i)] = np.load(input_training_data_dir + "N_Cifar10_"+str(n_monitors)+"M/" + "32_M" + str(i) + ".npy")
			data["N"]["56"]["M"+str(i)] = np.load(input_training_data_dir + "N_Cifar10_"+str(n_monitors)+"M/" + "56_M" + str(i) + ".npy")
		if a_type == "A":
			for i in range(1,n_monitors+1):
				data["A"]["20"]["M"+str(i)] = np.load(input_training_data_dir + "A_Cifar10_"+str(n_monitors)+"M/" + "20_M" + str(i) + ".npy")
				data["A"]["32"]["M"+str(i)] = np.load(input_training_data_dir + "A_Cifar10_"+str(n_monitors)+"M/" + "32_M" + str(i) + ".npy")
				data["A"]["56"]["M"+str(i)] = np.load(input_training_data_dir + "A_Cifar10_"+str(n_monitors)+"M/" + "56_M" + str(i) + ".npy")
		elif a_type == "Ai":
			for i in range(1,n_monitors+1):
				data["A"]["20"]["M"+str(i)] = np.load(input_training_data_dir + "Ai_Cifar10_"+str(n_monitors)+"M/" + "20_M" + str(i) + ".npy")
				data["A"]["32"]["M"+str(i)] = np.load(input_training_data_dir + "Ai_Cifar10_"+str(n_monitors)+"M/" + "32_M" + str(i) + ".npy")
				data["A"]["56"]["M"+str(i)] = np.load(input_training_data_dir + "Ai_Cifar10_"+str(n_monitors)+"M/" + "56_M" + str(i) + ".npy")
    
	elif mode == "inference":
		data={}
		data["20"] = {}
		data["32"] = {}
		data["56"] = {}
		for i in range(1,n_monitors+1):
			data["20"]["M"+str(i)] = np.load(input_inference_data_dir + "20_M" + str(i) + ".npy")
			data["32"]["M"+str(i)] = np.load(input_inference_data_dir + "32_M" + str(i) + ".npy")
			data["56"]["M"+str(i)] = np.load(input_inference_data_dir + "56_M" + str(i) + ".npy")
		
	else:
		pass

	return data

def load_ch_param (n_monitors): 
    data_temp = {}
    data_temp["20"] = {}
    data_temp["32"] = {}
    data_temp["56"] = {}
    data={}
    data["20"]={}
    data["32"]={}
    data["56"]={}
    data["20"]["M1"]={}
    data["20"]["M2"]={}
    data["20"]["M3"]={}
    data["32"]["M1"]={}
    data["32"]["M2"]={}
    data["32"]["M3"]={}
    data["56"]["M1"]={}
    data["56"]["M2"]={}
    data["56"]["M3"]={}
    for i in range(1,n_monitors+1):

        data_temp["20"]["M"+str(i)] = open(output_channels_normalization_dir + "20_M" + str(i) + ".txt",'r').read().splitlines()
        data_temp["32"]["M"+str(i)] = open(output_channels_normalization_dir + "32_M" + str(i) + ".txt",'r').read().splitlines()
        data_temp["56"]["M"+str(i)] = open(output_channels_normalization_dir + "56_M" + str(i) + ".txt",'r').read().splitlines()

    for cnn in data_temp :
        for monitor_layer in data_temp[cnn]:
            for norm_param in data_temp[cnn][monitor_layer]:
                index=norm_param.split(':')
                data[cnn][monitor_layer][index[0]]=float(index[1])

    return data


def load_ch_comp(n_monitors):
    data={}
    data["20"]={}
    data["32"]={}
    data["56"]={}

    for i in range(1,3+1):
        data["20"]["M"+str(i)] = pickle.load(open(output_channels_compression_dir + "20_M"+ str(i)+".pkl", "rb"))
        data["32"]["M"+str(i)] = pickle.load(open(output_channels_compression_dir + "32_M"+ str(i)+".pkl", "rb"))
        data["56"]["M"+str(i)] = pickle.load(open(output_channels_compression_dir + "56_M"+ str(i)+".pkl", "rb"))
    return data

def compress_dataset(dataset, reuse_parameters, compression_parameters_in, n_components_in):
	compressed_dataset = dataset.copy()
	compression_parameters = None

	if reuse_parameters == 0:
		compression_parameters = PCA(n_components=n_components_in)
		compressed_dataset = compression_parameters.fit_transform(compressed_dataset)
		columns = []
		for i in range(0, n_components_in):
			columns.append("f"+ str(i))
		compressed_dataset = pd.DataFrame(data=compressed_dataset, columns=columns)
	else:
		compressed_dataset = compression_parameters_in.transform(compressed_dataset)
		columns = []
		for i in range(0, n_components_in):
			columns.append("f"+ str(i))
		compressed_dataset = pd.DataFrame(data=compressed_dataset, columns=columns)


	return compressed_dataset, compression_parameters

def get_intervals(timeseries):

	intervals = {}
	
	columns = list(timeseries.columns)
	for column in columns:
		intervals[column] = [9999999999, -9999999999]
	for column in timeseries:
		temp_max = timeseries[column].max()
		temp_min = timeseries[column].min()
		if intervals[column][0] > temp_min:
			intervals[column][0] = temp_min
		if intervals[column][1] < temp_max:
			intervals[column][1] = temp_max

	return intervals

def normalize_dataset(dataset, reuse_parameters, normalization_parameters_in):
	
	normalized_dataset = dataset.copy() 
	normalization_parameters = {}

	if reuse_parameters == 0:
		if normalization_technique == "zscore":
			for column in normalized_dataset:
				column_values = normalized_dataset[column].values
				column_values_mean = np.mean(column_values)
				column_values_std = np.std(column_values)
				if column_values_std == 0:
					column_values_std = 1
				column_values = (column_values - column_values_mean)/column_values_std
				normalized_dataset[column] = column_values
				normalization_parameters[column+"_mean"] = column_values_mean
				normalization_parameters[column+"_std"] = column_values_std
		elif normalization_technique == "min-max":
			column_intervals = get_intervals(dataset)
			for column in normalized_dataset:
				column_data = normalized_dataset[column].tolist()
				intervals = column_intervals[column]
				if intervals[0] != intervals[1]:
					for idx,sample in enumerate(column_data):
						column_data[idx] = (sample-intervals[0])/(intervals[1]-intervals[0])
				normalized_dataset[column] = column_data
			for column in column_intervals:
				normalization_parameters[column+"_min"] = column_intervals[column][0]
				normalization_parameters[column+"_max"] = column_intervals[column][1]
	else:
		if normalization_technique == "zscore":
			for label in normalized_dataset:
				mean = normalization_parameters_in[label+"_mean"]
				std = normalization_parameters_in[label+"_std"]
				parameter_values = normalized_dataset[label].values
				parameter_values = (parameter_values - float(mean))/float(std)
				normalized_dataset[label] = parameter_values
		elif normalization_technique == "min-max":
			for label in normalized_dataset:
				min = normalization_parameters_in[label+"_min"]
				max = normalization_parameters_in[label+"_max"]
				parameter_values = normalized_dataset[label].values
				if min != max:
					for idx,sample in enumerate(parameter_values):
						parameter_values[idx] = (sample-min)/(max-min)
				normalized_dataset[label] = parameter_values
	
	return normalized_dataset, normalization_parameters

def split_data(data):

	train_data = data["N"]
	anomalous_data = data["A"]
	test_data = {}
	

	for cnn in train_data:
		test_data[cnn] = {}
		for monitor_layer in train_data[cnn]:
			train_data[cnn][monitor_layer], test_data[cnn][monitor_layer] = train_test_split(train_data[cnn][monitor_layer],test_size=test_percentage,shuffle=True)
	return train_data, test_data, anomalous_data

def normalize_channels(data, reuse_parameters, channels_normalization_parameters_in):

	channels_normalization_parameters = {}
	normalized_channels = {}

	if reuse_parameters == 0:
		for cnn in data:
			channels_normalization_parameters[cnn] = {}
			normalized_channels[cnn] = {}
			for monitor_layer in data[cnn]:
				channels_normalization_parameters[cnn][monitor_layer] = None
				normalized_channels[cnn][monitor_layer] = None
				rows = []
				for idx,sample in enumerate(data[cnn][monitor_layer]):
					for channel_idx,channel in enumerate(sample):
						for row in sample[channel_idx]:
							rows.append(row)

				temp_columns = list(range(0,len(rows[0])))
				temp_columns = [str(x) for x in temp_columns]
				temp = pd.DataFrame(columns=temp_columns, data=rows)
				normalized_channels[cnn][monitor_layer], channels_normalization_parameters[cnn][monitor_layer] = normalize_dataset(temp, reuse_parameters, None)
				


		return normalized_channels, channels_normalization_parameters

	else:
		for cnn in data:
			for monitor_layer in data[cnn]:
				temp_copy = data[cnn][monitor_layer]
				for idx,sample in enumerate(data[cnn][monitor_layer]):
					rows = []
					for channel_idx,channel in enumerate(sample):
						for row in sample[channel_idx]:
							rows.append(row)
				
					temp_columns = list(range(0,len(rows[0])))
					temp_columns = [str(x) for x in temp_columns]
					temp = pd.DataFrame(columns=temp_columns, data=rows)
					normalized_rows, ignore = normalize_dataset(temp, reuse_parameters, channels_normalization_parameters_in[cnn][monitor_layer])
					for channel_idx, channel in enumerate(sample):
						data[cnn][monitor_layer][idx][channel_idx] = normalized_rows.iloc[channel_idx*len(rows[0]):(channel_idx+1)*len(rows[0])].to_numpy()
				
		return data, None

def compress_channels(data, reuse_parameters, channels_compression_parameters_in):

	channels_compression_parameters = {}
	compressed_channels = {}

	if reuse_parameters == 0:
		for cnn in data:
			channels_compression_parameters[cnn] = {}
			compressed_channels[cnn] = {}
			for monitor_layer in data[cnn]:
				channels_compression_parameters[cnn][monitor_layer] = None
				compressed_channels[cnn][monitor_layer] = None
				rows = []
				for idx,sample in enumerate(data[cnn][monitor_layer]):
					for channel_idx,channel in enumerate(sample):
						for row in sample[channel_idx]:
							rows.append(row)
				compressed_channels[cnn][monitor_layer], channels_compression_parameters[cnn][monitor_layer] = compress_dataset(rows, reuse_parameters, None, 1)

		return compressed_channels, channels_compression_parameters
				
	else:
		for cnn in data:
			compressed_channels[cnn] = {}
			for monitor_layer in data[cnn]:
				compressed_channels[cnn][monitor_layer] = []
				for idx,sample in enumerate(data[cnn][monitor_layer]):
					rows = []
					for channel_idx,channel in enumerate(sample):
						for row in sample[channel_idx]:
							rows.append(row)
					compressed_channel, ignore = compress_dataset(rows, reuse_parameters, channels_compression_parameters_in[cnn][monitor_layer], 1)
					compressed_channels[cnn][monitor_layer].append(compressed_channel)

				concatenated_compressed_channels = []
				for sample in compressed_channels[cnn][monitor_layer]:
					concatenated_compressed_channels.append(list(sample.values))
				compressed_channels[cnn][monitor_layer] = pd.DataFrame(columns=list(range(len(concatenated_compressed_channels[0]))), data=concatenated_compressed_channels)
		
		return compressed_channels, None

def save_preprocessing_parameters(channels_normalization_parameters, channels_compression_parameters, final_compression_parameters):

	if channels_normalization_parameters != None:
		for cnn in channels_normalization_parameters:
			for monitor_layer in channels_normalization_parameters[cnn]:
				file = open(output_channels_normalization_dir + cnn + "_" + monitor_layer + ".txt", "w")
				for idx,normalization_parameter in enumerate(channels_normalization_parameters[cnn][monitor_layer]):
					if idx<len(channels_normalization_parameters[cnn][monitor_layer])-1:
						file.write(normalization_parameter + ":" + str(channels_normalization_parameters[cnn][monitor_layer][normalization_parameter]) + "\n")
					else:
						file.write(normalization_parameter + ":" + str(channels_normalization_parameters[cnn][monitor_layer][normalization_parameter]))
				file.close()
	if channels_compression_parameters != None:
		for cnn in channels_compression_parameters:
			for monitor_layer in channels_compression_parameters[cnn]:
				pickle.dump(channels_compression_parameters[cnn][monitor_layer], open(output_channels_compression_dir + cnn + "_" + monitor_layer + ".pkl","wb"))
	if final_compression_parameters != None:
		for cnn in final_compression_parameters:
			for monitor_layer in final_compression_parameters[cnn]:
				pickle.dump(final_compression_parameters[cnn][monitor_layer], open(output_final_compression_dir + cnn + "_" + monitor_layer + ".pkl","wb"))
	return None
	
def save_dataset(dataset, data_type, cnn, monitor_layer):
	if data_type == "TR":
		dataset.to_csv(output_train_data_dir + cnn + "_" + monitor_layer + ".csv", index=False)
	elif data_type == "TST":
		dataset.to_csv(output_test_data_dir + cnn + "_" + monitor_layer + ".csv", index=False)
	elif data_type == "AN":
		dataset.to_csv(output_anomalous_data_dir + cnn + "_" + monitor_layer + ".csv", index=False)
	elif data_type == "INF":
		dataset.to_csv(output_inference_data_dir+ cnn + "_" + monitor_layer + ".csv", index=False)
	return None

try:
	debug_mode = int(sys.argv[1])
	mode = sys.argv[2]
	if mode == "training":
		a_type = sys.argv[3]
		n_monitors = int(sys.argv[4])
		test_percentage = float(sys.argv[5])
		normalize = sys.argv[6]
		final_compression = sys.argv[7]
		if final_compression == "Y":
			n_components_final = int(sys.argv[8])
			
			
	elif mode == "inference":
		n_monitors = int(sys.argv[4])
		normalize = sys.argv[6]
		final_compression = sys.argv[7]
	else:
		print("Incorrect mode")
		sys.exit()

except IndexError:
	if mode == "":
		print("Not enough input arguments. Please, insert the input arguments")
		sys.exit()


if mode == "training":
	channels_normalization_parameters = None
	channels_compression_parameters = None
	final_compression_parameters = None

	data = load_data(mode, a_type, n_monitors)

	if debug_mode == 1:
		# Reduced dataset for debug purposes (please, set test_percentage=0.5 and REDUCED_LEN to an even number for everything to work correctly)
		REDUCED_LEN = 512
		for i in range(1,n_monitors+1):
			data["N"]["20"]["M"+str(i)] = data["N"]["20"]["M"+str(i)][0:REDUCED_LEN]
			data["N"]["32"]["M"+str(i)] = data["N"]["32"]["M"+str(i)][0:REDUCED_LEN]
			data["N"]["56"]["M"+str(i)] = data["N"]["56"]["M"+str(i)][0:REDUCED_LEN]
			data["A"]["20"]["M"+str(i)] = data["A"]["20"]["M"+str(i)][0:int(REDUCED_LEN/2)]
			data["A"]["32"]["M"+str(i)] = data["A"]["32"]["M"+str(i)][0:int(REDUCED_LEN/2)]
			data["A"]["56"]["M"+str(i)] = data["A"]["56"]["M"+str(i)][0:int(REDUCED_LEN/2)]


	train_data, test_data, anomalous_data = split_data(data)
	if normalize == "Y":
		ignore, channels_normalization_parameters = normalize_channels(train_data, 0, None)
		train_data, ignore = normalize_channels(train_data, 1, channels_normalization_parameters)
		test_data, ignore = normalize_channels(test_data, 1, channels_normalization_parameters)
		anomalous_data, ignore = normalize_channels(anomalous_data, 1, channels_normalization_parameters)

	ignore, channels_compression_parameters = compress_channels(train_data, 0, None)
	train_data, ignore = compress_channels(train_data, 1, channels_compression_parameters)
	test_data, ignore = compress_channels(test_data, 1, channels_compression_parameters)
	anomalous_data, ignore = compress_channels(anomalous_data, 1, channels_compression_parameters)

	if final_compression == "Y":
		final_compression_parameters = {}
		for cnn in train_data:
			final_compression_parameters[cnn] = {}
			for monitor_layer in train_data[cnn]:
				train_data[cnn][monitor_layer], final_compression_parameters[cnn][monitor_layer] = compress_dataset(train_data[cnn][monitor_layer], 0, None, n_components_final)
				test_data[cnn][monitor_layer], ignore = compress_dataset(test_data[cnn][monitor_layer], 1, final_compression_parameters[cnn][monitor_layer], n_components_final)
				anomalous_data[cnn][monitor_layer], ignore = compress_dataset(anomalous_data[cnn][monitor_layer], 1, final_compression_parameters[cnn][monitor_layer], n_components_final)

	if debug_mode == 1:
		e_dist = {}
		for cnn in ["20","32","56"]:
			e_dist[cnn] = {}
			for i in range(1,n_monitors+1):
				e_dist[cnn]["M"+str(i)] = {}
				e_dist[cnn]["M"+str(i)]["tr_tst"] = np.linalg.norm(train_data[cnn]["M"+str(i)]-test_data[cnn]["M"+str(i)])
				e_dist[cnn]["M"+str(i)]["tr_an"] = np.linalg.norm(train_data[cnn]["M"+str(i)]-anomalous_data[cnn]["M"+str(i)])
				print("The Euclidean distance between the train set and the test set of cnn " + cnn + " for monitor M" + str(i) + " is: " + str(e_dist[cnn]["M"+str(i)]["tr_tst"]))
				print("The Euclidean distance between the train set and the anomalous set of cnn " + cnn + " for monitor M" + str(i) + " is: " + str(e_dist[cnn]["M"+str(i)]["tr_an"]))
	
	
	save_preprocessing_parameters(channels_normalization_parameters, channels_compression_parameters, final_compression_parameters)

	for cnn in train_data:
		for monitor_layer in train_data[cnn]:
			if final_compression != "Y":
				temp_columns = list(range(0,train_data[cnn][monitor_layer].shape[1]))
				temp = pd.DataFrame(data=train_data[cnn][monitor_layer], columns=temp_columns)
				for column in temp:
					temp[column] = temp[column].str[0]
				save_dataset(temp, "TR", cnn, monitor_layer)
			else:
				save_dataset(train_data[cnn][monitor_layer], "TR", cnn, monitor_layer)

	for cnn in test_data:
		for monitor_layer in test_data[cnn]:
			if final_compression != "Y":
				temp_columns = list(range(0,test_data[cnn][monitor_layer].shape[1]))
				temp = pd.DataFrame(data=test_data[cnn][monitor_layer], columns=temp_columns)
				for column in temp:
					temp[column] = temp[column].str[0]
				save_dataset(temp, "TST", cnn, monitor_layer)
			else:
				save_dataset(test_data[cnn][monitor_layer], "TST", cnn, monitor_layer)

	for cnn in anomalous_data:
		for monitor_layer in anomalous_data[cnn]:
			if final_compression != "Y":
				temp_columns = list(range(0,anomalous_data[cnn][monitor_layer].shape[1]))
				temp = pd.DataFrame(data=anomalous_data[cnn][monitor_layer], columns=temp_columns)
				for column in temp:
					temp[column] = temp[column].str[0]
				save_dataset(temp, "AN", cnn, monitor_layer)
			else:
				save_dataset(anomalous_data[cnn][monitor_layer], "AN", cnn, monitor_layer)

elif mode == "inference":
	data = load_data(mode, a_type, n_monitors)
	param =load_ch_param(n_monitors)
	channels_compression_parameters=load_ch_comp(n_monitors)
	if normalize == "Y":
		data, ignore = normalize_channels(data, 1, param)
	
	data, ignore = compress_channels(data, 1, channels_compression_parameters)
	for cnn in data:
		for monitor_layer in data[cnn]:
			if final_compression != "Y":
				temp_columns = list(range(0,data[cnn][monitor_layer].shape[1]))
				temp = pd.DataFrame(data=data[cnn][monitor_layer], columns=temp_columns)
				for column in temp:
					temp[column] = temp[column].str[0]
				save_dataset(temp, "INF", cnn, monitor_layer)
			else:
				save_dataset(data[cnn][monitor_layer], "INF", cnn, monitor_layer)
