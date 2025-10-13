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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []

for dataset_name in UNIVARIATE_DATASET_NAMES:
    print(dataset_name)
    train_loader, test_loader, train_shape, test_shape, ucr_nb_classes = ucr_loader(dataset_name, batch_size=128)
    
    model = Classifier_INCEPTION_with_Jitter(input_shape=train_shape, nb_classes=ucr_nb_classes,device=device).to(device)
    model.load_state_dict(torch.load(f'/Project/Yi/defense/OUTPUT/jitter0.750.6/{dataset_name}_jitter_model_weights.pth'))

    model.eval()  

    attack = FGSM(model, eps=0.1, device=device)
    #attack = BIM(model, eps=0.1, beta=0.0005, num_iters=1000, device=device)
    #attack = PGD(model, eps_init=0.001, eps=0.1, beta=0.0005, num_iters=1000, device=device)
    total = 0
    successful_attacks = 0
    l2_distances = []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs_natural = model(inputs)
        preds_natural = torch.argmax(outputs_natural, dim=1)
        adv_data = attack.generate(inputs)

        outputs_adv = model(adv_data)
        preds_adv = torch.argmax(outputs_adv, dim=1)
        attack_success = preds_adv != preds_natural
        successful_attacks += attack_success.sum().item()
        successful_adv_data = adv_data[attack_success]

        if successful_adv_data.nelement() > 0:
            successful_inputs = inputs[attack_success]

            l2_distance = torch.norm(successful_adv_data - successful_inputs, p=2, dim=[1, 2])
 
            total_l2_distance = torch.sum(l2_distance).item()
            l2_distances.append(total_l2_distance)  
            total += labels.size(0)

    attack_success_rate = successful_attacks / total if total else 0
    average_l2_distance = sum(l2_distances) / successful_attacks if successful_attacks else 0

    results.append([dataset_name, attack_success_rate, average_l2_distance])

results_df = pd.DataFrame(results, columns=['Dataset Name', 'Attack Success Rate', 'Average L2 Distance'])
csv_file_path = '/Project/Yi/defense/OUTPUT/jitter0.750.6/fgsm_resultsj0.750.6.csv'
results_df.to_csv(csv_file_path, index=False)

print("Results saved to:", csv_file_path)
