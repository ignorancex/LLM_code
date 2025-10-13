import os
print(os.getcwd())
import sys
sys.path.append('/Project/Project/Yi/defense')
from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Model.inception import *
from CODE.Utils.augmentation import *

results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for dataset_name in UNIVARIATE_DATASET_NAMES:

    train_loader, test_loader, train_shape, test_shape, ucr_nb_classes = ucr_loader(dataset_name, batch_size=64)
    
    model = Classifier_INCEPTION_with_gaussian_noise(input_shape=train_shape, nb_classes=ucr_nb_classes, device=device).to(device)
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
    torch.save(model.state_dict(), f"/Project/Project/Yi/defense/OUTPUT/gaussian_noise/{dataset_name}_gaussian_noise_model_weights.pth")


    model = Classifier_INCEPTION_with_gaussian_noise(input_shape=train_shape, nb_classes=ucr_nb_classes,device=device).to(device)
    model.load_state_dict(torch.load(f"/Project/Project/Yi/defense/OUTPUT/gaussian_noise/{dataset_name}_gaussian_noise_model_weights.pth"))
    
    model.eval()  
    dataset_results = []  
    for i in range(5):
        
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

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
csv_file_path = '/Project/Project/Yi/defense/OUTPUT/gaussian_noise/resultsgn.csv'

if os.path.exists(csv_file_path):
    results_df.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(csv_file_path, mode='w', header=True, index=False)

print("Results have been saved to CSV.")
