import os
print(os.getcwd())
import sys
sys.path.append('/Project/Yi/defense')
from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Model.inception import *
from CODE.Utils.augmentation import *
from CODE.attack.methods import *
from CODE.Model.models import *

class EnsembleModel(nn.Module):
    def __init__(self, models, device, dataset_name):
        super(EnsembleModel, self).__init__()
        self.models = models  # models should be a dictionary
        self.device = device
        self.dataset_name = dataset_name
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        inputs = inputs.to(self.device)
        average_output = sum(self.models[name](inputs) for name in self.models) / len(self.models)
        #output = self.softmax(average_output)
        return average_output.requires_grad_(True)
    def print_dataset_name(self):
        print(f"Using ensemble model for dataset: {self.dataset_name}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_file_path = '/Project/Yi/defense/OUTPUT/ensemble2/pgd_resultensembleres.csv'

file_exists = False

try:
    with open(csv_file_path, 'r'):
        file_exists = True
except FileNotFoundError:
    file_exists = False

columns = ['Dataset Name', 'Robust Accuracy']

if not file_exists:
    results_df = pd.DataFrame(columns=columns)
    results_df.to_csv(csv_file_path, mode='w', index=False)
    file_exists = True 

model_paths = {
    'gn': '/Project/Yi/defense/OUTPUT/res_gn/{dataset_name}_1_model_weights.pth',
    'j': '/Project/Yi/defense/OUTPUT/res_jitter/{dataset_name}_1_model_weights.pth',
    'rz': '/Project/Yi/defense/OUTPUT/res_rz/{dataset_name}_1_model_weights.pth',
    'sz': '/Project/Yi/defense/OUTPUT/res_sz/{dataset_name}_1_model_weights.pth',
    'sts': '/Project/Yi/defense/OUTPUT/res_sts/{dataset_name}_1_model_weights.pth',
    '0': '/Project/Yi/defense/OUTPUT/resnet/{dataset_name}_0_model_weights.pth'
}

for dataset_name in UNIVARIATE_DATASET_NAMES2:
    print(f"Loading models for dataset: {dataset_name}")
    train_loader, test_loader, train_shape, test_shape, ucr_nb_classes = ucr_loader(dataset_name, batch_size=64)

    models = {
        "gn": ClassifierResNet18_with_gaussian_noise(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device),
        "j": ClassifierResNet18_with_Jitter(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device),
        "rz": ClassifierResNet18_with_RandomZero(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device),
        "sz": ClassifierResNet18_with_SegmentZero(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device),
        "sts": ClassifierResNet18_with_smooth_time_series(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device),
        "0": ClassifierResNet18(input_shape=1, nb_classes=ucr_nb_classes).to(device)
    }

    for model_key in models:
        models[model_key].load_state_dict(torch.load(model_paths[model_key].format(dataset_name=dataset_name)))

    for model in models.values():
        model.eval()

    # Initialize the ensemble model
    ensemble_model = EnsembleModel(models, device, dataset_name)
    
    # Define the attacker using the ensemble model
    attack = PGD(ensemble_model, eps_init=0.001, eps=0.1, beta=0.0005, num_iters=1000, device=device)

    print(f"Evaluating dataset: {dataset_name}")
    total_batches = len(test_loader)  
    batch_counter = 1  
    #total = 0
    #successful_attacks = 0
    #l2_distances = []
    all_labels = []
    all_predictions = []
    for inputs, labels in test_loader:

        inputs, labels = inputs.to(device), labels.to(device)

        #for model in models.values():
            #model.train()

        #outputs_natural = model(inputs)
        #preds_natural = torch.argmax(outputs_natural, dim=1)
        adv_data = attack.generate(inputs)
        #for model in models.values():
            #model.eval()

        outputs_adv = ensemble_model(adv_data)
        _, predicted = torch.max(outputs_adv.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        '''
        preds_adv = torch.argmax(outputs_adv, dim=1)
        
        attack_success = preds_adv != labels

        successful_attacks += attack_success.sum().item()

        successful_adv_data = adv_data[attack_success]
        
        if successful_adv_data.nelement() > 0:
            successful_inputs = inputs[attack_success]

            l2_distance = torch.norm(successful_adv_data - successful_inputs, p=2, dim=[1, 2])
 
            total_l2_distance = torch.sum(l2_distance).item()
            l2_distances.append(total_l2_distance)  
            total += labels.size(0)
        '''
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Processing batch {batch_counter} of {total_batches} for dataset {dataset_name}")
        batch_counter += 1 

    accuracy = accuracy_score(all_labels, all_predictions)
    
    #average_l2_distance = sum(l2_distances) / successful_attacks if successful_attacks else 0
    result = [[dataset_name, accuracy]]

    results_df = pd.DataFrame(result, columns=columns)
    results_df.to_csv(csv_file_path, mode='a', header=not file_exists, index=False)

    print(f"Results for {dataset_name} saved to:", csv_file_path)

print("All results have been saved.")
