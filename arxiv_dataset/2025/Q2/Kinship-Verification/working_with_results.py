import os
import pandas as pd
import matplotlib.pyplot as plt

def read_txt(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
    file.close()
    return lines

def get_total_acc(result: list[str]):
    ls = []
    for i in result:
        ls.append(i.split('\t'))
        ls[-1] = [ls[-1][0], float(ls[-1][2])]
    return ls

def get_directories(folder_path, idx=-1) -> list[str]:
    # List all directories recursively
    accs_directories = []
    for _, dirs, _ in os.walk(folder_path):
        for dir_name in dirs:
            for _, _, files in os.walk(os.path.join(folder_path, dir_name)):
                accs_directories.append(os.path.join(
                    folder_path, dir_name, files[idx]))
    return accs_directories

def import_all_configs():
    accs_directories = get_directories(folder_path='Results', idx=0)

    results = {}
    for directory in accs_directories:
        key = directory.split('\\')[-1]
        config = read_txt(directory)
        dict_config = {}
        for i in range(len(config)):
            config[i] = config[i].split(':')
            config[i] = [config[i][0].strip(), config[i][1][1:]]
            try:
                config[i][1] = int(config[i][1]) 
            except ValueError:
                try:
                    config[i][1] = float(config[i][1])
                except ValueError:
                    config[i][1] = bool(config[i][1])
            
            dict_config[config[i][0]] = config[i][1]                
              
        results[key] = dict_config
        
    return results

def get_best_configs():
    accs_directories = get_directories(folder_path='Results')
    x = get_directories(folder_path='Results', idx=0)
    results = {}
    
    for directory in accs_directories:
        key = x[accs_directories.index(directory)].split('\\')[2]
        try:
            results[key] = get_total_acc(read_txt(directory))
        except:
            print(directory)
        
    bests = {
        'father_son': [0, ''],
        'mother_son': [0, ''],
        'mother_dau': [0, ''],
        'father_dau': [0, ''],
    }

    for key in results.keys():
        x = results[key]
        for i in x:
            if i[1] > bests[i[0]][0]:
                bests[i[0]][0] = i[1]
                bests[i[0]][1] = key

    for key in results.keys():
        x = results[key]
        for i in x:
            if i[1] == bests[i[0]][0] and key not in bests[i[0]]:
                bests[i[0]].append(key)


    for key in bests.keys():
        print(key, '\t', bests[key][0])
        for i in bests[key][1:]:
            print('\t', i)
    print('Total Acc =',
          f'{sum([bests[key][0]  for key in bests.keys()])/4:.2f}')
   

    all = {
        'father_son': {},
        'mother_son': {},
        'mother_dau': {},
        'father_dau': {},
    }

    for key in results.keys():
        x = results[key]
        for i in x:
            all[i[0]][key] = i[1]
    
    unique_configs = []

    for key, value in all.items():
        #print(key)
        all[key] = {k: all[key][k] for k in sorted(all[key], key=all[key].get, reverse=True)}
        all[key] = {k: all[key][k] for k in list(all[key].keys())[:5]}

        for k, v in all[key].items():
            if k not in unique_configs:
                unique_configs.append(k)
            #print('\t' , k, '\t', v)

        #print('\n')
    
    #print(len(unique_configs))  
    return unique_configs

def roc_curve():
    # Load the saved CSV file
    df = pd.read_csv("Results/ccl_decay-1.03_fc1-256_fc2-8_sep-True_outdim-600/detail_father_dau.csv")

    # Compute TPR and FPR
    df["TPR"] = df["TP"] / (df["TP"] + df["FN"])
    df["FPR"] = df["FP"] / (df["FP"] + df["TN"])

    # Plot ROC Curve for each fold
    plt.figure(figsize=(8, 8))

    for fold, fold_df in df.groupby("fold_num"):
        if fold == 1:
            plt.plot(fold_df["FPR"], fold_df["TPR"], marker='o', label=f"Fold {fold}")

    # Plot random guess line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curves for Different Folds")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    
    # roc_curve()
    #get_best_configs()
    
    
    configs = import_all_configs()
    best_configs = get_best_configs()
    
    # best_config_details = {}
    
    # for i in best_configs:
    #     best_config_details[i] = configs[i]
    
    # keys_i_need = ['CCLdecayrate', 'FC1', 'FC2', 'Seperate', 'Outputdim']
    
    # for k, v in best_config_details.items():
    #     for key, value in v.items():
    #         if key in keys_i_need:
    #             print(f"'{key}'", ':', value, ',')
    #     print('\n')

        
