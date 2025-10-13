from core.generation_runner import TaskGenerationPromptRunner
from core.classification_runner import TaskClassificationPromptRunner
from utils.helper_functions import generate_id
from utils.infeasible_reasons import infeasible_reasons
import pandas as pd, random, os, argparse

def generate_tasks(type_of_task:str, prompt_type:str, number_of_tasks: int, model:str, type_of_self_knowledge:str):
    df = pd.DataFrame(columns=['ID', 'Task', 'LLM','Task Category (Generation)', 'Type of Self-Knowledge (Generation)', 'Reason for Incapacity (Generation)'])
    for i in range(number_of_tasks):
        # try:
            if type_of_task.lower() == 'infeasible':  
                infeasibility_reason = random.choice(list(infeasible_reasons[type_of_self_knowledge].items()))
            else:
                infeasibility_reason = None
            prompt_runner = TaskGenerationPromptRunner(task_type=type_of_task,prompt_type=prompt_type,model=model, type_of_self_knowledge=type_of_self_knowledge,infeasibility_reason=infeasibility_reason)
            prompt_runner.get_response()
            task = prompt_runner.response
            rows = []
            rows.append({
                    'ID': generate_id(),
                    'Task': task['task'],
                    'LLM': model,
                    'Task Category (Generation)': type_of_task.capitalize(),
                    'Type of Self-Knowledge (Generation)': type_of_self_knowledge,
                    'Reason for Incapacity (Generation)': infeasibility_reason[0] if infeasibility_reason else 'N/A'
            })
            new_data = pd.DataFrame(rows)
            if os.path.exists(f'generation_results/{model}_results.xlsx'):
                df = pd.read_excel(f'generation_results/{model}_results.xlsx')
                df = pd.concat([df, new_data], ignore_index=True)
            else:
                df = new_data
            df.to_excel(f'generation_results/{model}_results.xlsx', index=False, engine='xlsxwriter')
        # except:
        #     print(f'Error in generating task {i+1}')
            # print(f'{model} - {type_of_self_knowledge} - {(i+1)} {type_of_task} tasks generated')


def generate_random_subset(file_path:str, num_samples:int):
    df = pd.read_excel(file_path)
    categories = ['Feasible', 'Infeasible']  
    df = pd.concat([
        df[df['Task Category (Generation)'] == category].sample(n=num_samples, random_state=42)
        for category in categories
    ])
    df = df.sample(frac=1, random_state=68).reset_index(drop=True)
    random_sample_path = file_path.replace('.xlsx', '_random_sample.xlsx')
    df.to_excel(random_sample_path, index=False, engine='xlsxwriter')
    return random_sample_path


def classify_tasks(file_path:str, prompt_type:str, model:str):
    df = pd.read_excel(file_path)
    if 'Task Category (Classification)' not in df.columns:
        df['Task Category (Classification)'] = ''
    if 'Type of Self-Knowledge (Classification)' not in df.columns:
        df['Type of Self-Knowledge (Classification)'] = ''
    for index, row in df.iterrows():
        try:
            task = row['Task']
            prompt_runner = TaskClassificationPromptRunner(prompt_type=prompt_type,task=task, model=model)
            prompt_runner.get_response()
            if 'task_class' in prompt_runner.response:
                df.at[index, 'Task Category (Classification)'] = str(prompt_runner.response['task_class']).capitalize()
                if str(prompt_runner.response['task_class']).lower() == 'infeasible':
                    df.at[index, 'Reason for Incapacity (Classification)'] = str(prompt_runner.response['reason'])
                    df.at[index, 'Type of Self-Knowledge (Classification)'] = str(prompt_runner.response['type_of_self_knowledge']) 
                else:
                    df.at[index, 'Reason for Incapacity (Classification)'] = 'N/A'
                    df.at[index, 'Type of Self-Knowledge (Classification)'] = df.at[index, 'Type of Self-Knowledge (Generation)']
                print(f'{model} - {index+1} tasks classified')
                df.to_excel(f'classification_results/{model}_results.xlsx', index=False, engine='xlsxwriter')
            else:
                df.at[index, 'Task Category (Classification)'] = 'Error'
                print(f'Error in classifying task {index+1}')
                df.to_excel(f'classification_results/{model}_results.xlsx', index=False, engine='xlsxwriter')
        except Exception as e: 
            df.at[index, 'Task Category (Classification)'] = 'Error'
            df.at[index, 'Reason for Incapacity (Classification)'] = 'N/A'
            print(f'Error in classifying task {index+1} - {e}')
    print(f'{model} - All tasks classified')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and classify tasks")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Choose a command: generate or classify")

    generate_parser = subparsers.add_parser("generate", help="Generate tasks")
    generate_parser.add_argument("--model", type=str, help="select from ['gpt-4o-mini','gpt-4o','gemini-1.5-flash','claude-3-5-sonnet-latest','mistral-large-latest']", required=True)
    generate_parser.add_argument("--type_of_self_knowledge", type=str, help="select from ['Capability Boundary','Contextual Awareness','Identification of Ambiguity','Ethical Integrity','Temporal Perception']", required=True)
    generate_parser.add_argument("--type_of_task", type=str, help="select from ['feasible','infeasible']", required=True)
    generate_parser.add_argument("--prompt_type", type=str, help="select from ['vanilla','challenge_qap']", required=True)
    generate_parser.add_argument("--number_of_tasks", type=int, help="number of tasks to generate", required=True)

    classify_parser = subparsers.add_parser("classify", help="Classify tasks")
    classify_parser.add_argument("--model", type=str, help="select from ['gpt-4o-mini','gpt-4o','gemini-1.5-flash','claude-3-5-sonnet-latest','mistral-large-latest']", required=True)
    classify_parser.add_argument("--file_path", type=str, help="path to the file containing tasks to classify", required=True)
    classify_parser.add_argument("--prompt_type", type=str, help="select from ['vanilla','challenge_qap']", required=True)

    args = parser.parse_args()
    if args.command == "generate":
        generate_tasks(type_of_task=args.type_of_task, prompt_type=args.prompt_type, number_of_tasks=args.number_of_tasks, model=args.model, type_of_self_knowledge=args.type_of_self_knowledge)
    elif args.command == "classify":
        classify_tasks(file_path=args.file_path, prompt_type=args.prompt_type, model=args.model)