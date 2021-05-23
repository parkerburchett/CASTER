def dde_NN_config():
	"""
		Return a dictionary that is the details for the shape and other hyperparams used in the main Neural Network
	"""
	#draft 1
	config = {}
	config['batch_size'] = 256
	config['input_dim'] = 1722 # I think this refers the number of functional groups extracted -P
	config['batch_first'] = True
	config['num_class'] = 2 # binary classification problem
	config['LR'] = 1e-3 # this just says that learning rate is .001
	config['train_epoch'] = 3
	config['pretrain_epoch'] = 1
	config['recon_threshold'] = 0.0005 # change later #- p I don't know what this is

	config['encode_fc1_dim'] = 500  # encoder fc1
	config['encode_fc2_dim'] = 50  # encoder fc2
	config['decode_fc1_dim'] = 500  # decoder fc1
	config['decode_fc2_dim'] = config['input_dim']  # decoder reconstruction
	config['predict_dim'] = 1024 # for every layer
	config['predict_out_dim'] = 1 # predictor out # This is on of off. corrosponds to a 1 or 0 in the target variables. 
	config['lambda1'] = 1e-2  # L1 regularization coefficient
	config['lambda2'] = 1e-1  # L2 regulatization coefficient
	config['lambda3'] = 1e-5  # L2 regulatization coefficient
	config['reconstruction_coefficient'] = 1e-1  # 1e-2
	config['projection_coefficient'] = 1e-1  # 1e-2
	config['magnify_factor'] = 100
	return config