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

results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for dataset_name in UNIVARIATE_DATASET_NAMES2:
    start_time = time.time()  
    train_loader, test_loader, train_shape, test_shape, ucr_nb_classes = ucr_loader(dataset_name, batch_size=128)
    
    model = ClassifierResNet18(input_shape=1, nb_classes=ucr_nb_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    pgd = PGD_for_AT(model, eps_init=0.001, eps=0.1, beta=0.0005, num_iters=40, device=device)

    for epoch in range(1000):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_adv = pgd.generate(inputs, labels)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            
            #loss.backward()

            output_adv = model(inputs_adv)
            loss_adv = criterion(output_adv, labels)

            total_loss = loss + loss_adv
            total_loss.backward()

            optimizer.step()
        
        print(f"Epoch {epoch+1}: Loss = {loss.item()}, Adversarial Loss = {loss_adv.item()}")
    torch.save(model.state_dict(), f"/Project/Yi/defense/OUTPUT/res_AT40/{dataset_name}_AT040_model_weights.pth")
    end_time = time.time()
    total_time = end_time - start_time 

    model = ClassifierResNet18(input_shape=1, nb_classes=ucr_nb_classes).to(device)
    model.load_state_dict(torch.load(f"/Project/Yi/defense/OUTPUT/res_AT40/{dataset_name}_AT040_model_weights.pth"))
    
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
        'Average F1 Score': mean_f1,
        'Total Training Time': total_time
    })

results_df = pd.DataFrame(results)

csv_file_path = '/Project/Yi/defense/OUTPUT/res_AT40/resultsresAT040.csv'

if os.path.exists(csv_file_path):
    results_df.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(csv_file_path, mode='w', header=True, index=False)

print("Results have been saved to CSV.")
