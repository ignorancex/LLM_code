import os
print(os.getcwd())
import sys
sys.path.append('/Project/Yi/defense')
from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Model.inception import *
from CODE.Utils.augmentation import *
from CODE.Model.models import *

results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for dataset_name in UNIVARIATE_DATASET_NAMES2:

    train_loader, test_loader, train_shape, test_shape, ucr_nb_classes = ucr_loader(dataset_name, batch_size=256)
    '''
    model = Classifier_INCEPTION(input_shape=train_shape, nb_classes=ucr_nb_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1000):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    torch.save(model.state_dict(), f"/Project/Yi/defense/OUTPUT/0/{dataset_name}_0_model_weights.pth")
    '''
    modelgn = ClassifierResNet18_with_gaussian_noise(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device)
    modelgn.load_state_dict(torch.load(f'/Project/Yi/defense/OUTPUT/res_gn/{dataset_name}_1_model_weights.pth'))

    modelgn.eval() 
    modelj = ClassifierResNet18_with_Jitter(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device)
    modelj.load_state_dict(torch.load(f'/Project/Yi/defense/OUTPUT/res_jitter/{dataset_name}_1_model_weights.pth'))

    modelj.eval() 
    modelrz = ClassifierResNet18_with_RandomZero(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device)
    modelrz.load_state_dict(torch.load(f'/Project/Yi/defense/OUTPUT/res_rz/{dataset_name}_1_model_weights.pth'))

    modelrz.eval()
    modelsz = ClassifierResNet18_with_SegmentZero(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device)
    modelsz.load_state_dict(torch.load(f'/Project/Yi/defense/OUTPUT/res_sz/{dataset_name}_1_model_weights.pth'))

    modelsz.eval()
    modelsts = ClassifierResNet18_with_smooth_time_series(input_shape=1, nb_classes=ucr_nb_classes, device=device).to(device)
    modelsts.load_state_dict(torch.load(f'/Project/Yi/defense/OUTPUT/res_sts/{dataset_name}_1_model_weights.pth'))

    modelsts.eval()
    model0 = ClassifierResNet18(input_shape=1, nb_classes=ucr_nb_classes).to(device)
    model0.load_state_dict(torch.load(f'/Project/Yi/defense/OUTPUT/resnet/{dataset_name}_0_model_weights.pth'))

    model0.eval()
    dataset_results = []  
    for i in range(5):
        
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs_natural_gn = modelgn(inputs)
        
                outputs_natural_j = modelj(inputs)
            
                outputs_natural_rz = modelrz(inputs)
            
                outputs_natural_sz = modelsz(inputs)
        
                outputs_natural_sts = modelsts(inputs)
            
                outputs_natural_0 = model0(inputs)
                outputs_natural = (outputs_natural_gn+outputs_natural_j+outputs_natural_rz+outputs_natural_sz+outputs_natural_sts+outputs_natural_0)/6
                _, predicted = torch.max(outputs_natural.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        dataset_results.append({
            'Accuracy': accuracy * 100,  
            'F1 Score': f1
        })

    dataset_results_df = pd.DataFrame(dataset_results)
    mean_accuracy = dataset_results_df['Accuracy'].mean()
    mean_f1 = dataset_results_df['F1 Score'].mean()

    results.append({
        'Dataset': dataset_name,
        'Average Accuracy': mean_accuracy,
        'Average F1 Score': mean_f1
    })

results_df = pd.DataFrame(results)

csv_file_path = '/Project/Yi/defense/OUTPUT/ensemble/resultsensembleres.csv'

if os.path.exists(csv_file_path):
    results_df.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(csv_file_path, mode='w', header=True, index=False)

print("Results have been saved to CSV.")
