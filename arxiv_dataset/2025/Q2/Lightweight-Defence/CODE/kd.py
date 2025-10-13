import os
print(os.getcwd())
import sys
sys.path.append('/Project/Yi/defense')
from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Model.inception import *
from CODE.Utils.augmentation import *

def distillation_loss(output, labels, teacher_output, T=10.0, alpha=0.5):
    hard_loss = nn.CrossEntropyLoss()(output, labels)
    soft_loss = nn.KLDivLoss()(F.log_softmax(output/T, dim=1),
                               F.softmax(teacher_output/T, dim=1))
    return alpha * hard_loss + (1 - alpha) * soft_loss

def train_model(model, train_loader, criterion, optimizer, T=10.0, is_student=False, teacher_model=None, device=None):
    
    model.to(device)  
    for epoch in range(1000):
        if teacher_model:
            teacher_model.to(device) 
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if is_student and teacher_model:
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                loss = distillation_loss(outputs, labels, teacher_outputs)
            else:
                outputs = outputs / T  # Adjust logits by temperature before loss computation

                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    return model

results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for dataset_name in UNIVARIATE_DATASET_NAMES:
    start_time = time.time()  
    train_loader, test_loader, train_shape, test_shape, ucr_nb_classes = ucr_loader(dataset_name, batch_size=128)
    
    teacher_model = Classifier_INCEPTION_Logits(input_shape=train_shape, nb_classes=ucr_nb_classes).to(device)
    student_model = Classifier_INCEPTION_Logits(input_shape=train_shape, nb_classes=ucr_nb_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    teacher_optimizer = optim.Adam(teacher_model.parameters())
    student_optimizer = optim.Adam(student_model.parameters())

    #model.train()
    '''
    for inputs, labels in train_loader:
        
        teacher_model.train
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = teacher_model(inputs) #need to save the outputs of teacher
        loss = criterion(outputs, labels)
        '''
    # Train teacher model
    teacher_model = train_model(teacher_model, train_loader, criterion, teacher_optimizer, device=device)

    # Train student model with knowledge distillation
    

    student_model = train_model(student_model, train_loader, criterion, student_optimizer, is_student=True, teacher_model=teacher_model, device=device)
    
    #print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(student_model.state_dict(), f"/Project/Yi/defense/OUTPUT/kd/{dataset_name}_kd_model_weights.pth")
    end_time = time.time()  
    total_time = end_time - start_time 

    student_model = Classifier_INCEPTION(input_shape=train_shape, nb_classes=ucr_nb_classes).to(device)
    student_model.load_state_dict(torch.load(f"/Project/Yi/defense/OUTPUT/kd/{dataset_name}_kd_model_weights.pth"))
    # Testing student model
    student_model.eval()
    dataset_results = []  
    for i in range(5):
        
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student_model(inputs)
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

csv_file_path = '/Project/Yi/defense/OUTPUT/kd/resultskd.csv'

if os.path.exists(csv_file_path):
    results_df.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(csv_file_path, mode='w', header=True, index=False)

print("Results have been saved to CSV.")
