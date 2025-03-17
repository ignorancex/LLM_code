arity = 2
projection = [0,2,3]
drop = [1]
num_list = [0,1]
op_list = [2,3]

data_root = './data/'
model_path = './checkpoint/'
num_epochs = 1000
sgd_lr = 0.1; # learning rate of SGD
adam_lr = 0.001;  # learning rate of Adam
num_classes = 15; len_seq = 2*arity
sampling_epoch = 10; 
# setting projection and initilize labels
randomSeed = 986
numSamples = 10000

synopsis = f'arity={arity}_seed={randomSeed}_numSamples={numSamples}'
trainDatasetFile = f'{data_root}dataset_train_{synopsis}.pkl'
testDatasetFile = f'{data_root}dataset_test_{synopsis}.pkl'
random_labels_file = f'{data_root}random_labels_{synopsis}.pt'


