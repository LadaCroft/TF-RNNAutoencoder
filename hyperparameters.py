########## Hyperparameters #############
input_data = 'cos_mod'  # 'corn', 'ecg'
model_type = 'RNNAutoencoder'  # RNNAugmented
cell_type = 'LSTM'  # 'GRU'
steps_enc = 100
steps_dec = 20
batch_size = 400
num_layers = 1
num_hidden = 400
train_ratio = 0.9
learning_rate = 0.001
max_patience = 2000
max_epochs = 50000
optimizer_type = 'Adam'


LR = False
LR_part = 100

means = False
mean_part = 100


num_units = []
for layer in range(num_layers):
    num_units.append(num_hidden)


options = dict({'input_data': input_data,  # 'ecg','cos_mod'
                'model_type': model_type,  # 'RNNAugmented
                'cell_type': cell_type,  # 'GRU'
                'optimizer_type': optimizer_type,
                'train_ratio': train_ratio,
                'learning_rate': learning_rate,
                'max_patience': max_patience,
                'max_epochs': max_epochs,
                'batch_size': batch_size,
                'num_layers': num_layers,
                'num_hidden': num_hidden,
                'steps_enc': steps_enc,
                'steps_dec': steps_dec,
                'LR': LR,
                'LR_part': LR_part,
                'means': means,
                'mean_part': mean_part})
