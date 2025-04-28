import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

# Import prompts
from LLM.prompt import (
    Parse_user_input,
    Parse_user_input2,
    Summary_output,
    Determine_model,
    Analyzing_data
)

# Import AutoML modules
from AutoML.classification import main_sweet as classification_main
from AutoML.regression import main as regression_main

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

@dataclass
class ModelConfig:
    task_type: TaskType
    model_name: str
    data_columns: List[str]
    label_column: str
    hyperparameters: Dict
    file_path: str

class AutoPE:
    def __init__(
        self,
        inference_server_url: str,
        api_key: str,
        model_name: str,
        base_path: str = "."
    ):
        """Initialize AutoPE with LLM configuration."""
        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=inference_server_url,
            temperature=0.8,
        )
        self.base_path = Path(base_path)
        self.current_config = None

    def _read_file(self, file_path: str) -> pd.DataFrame:
        """Read data file with error handling."""
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")

    def _analyze_data_structure(self, user_input,  df: pd.DataFrame) -> Dict:
        """Analyze data structure using LLM."""
        data_description = df.describe().to_string()
        # print('df',df.head(3))
        columns_info = {
            "name": list(df.columns),
            "dtypes": [str(dtype) for dtype in df.dtypes],
            "sample": df.head(3).to_dict()
        }
        # print('columns_info:', columns_info)
        user_input += '\n The file contains the following columns: ' + ', '.join(df.columns)
        
        messages = [
            SystemMessage(content=Analyzing_data),
            HumanMessage(content=user_input)
        ]
        
        response = self.llm.invoke(messages)
        # print('response:', response)
        return json.loads(response.content)

    def _determine_task_type(self, df: pd.DataFrame, label_column: str) -> TaskType:
        """Determine whether the task is classification or regression."""
        unique_values = df[label_column].nunique()
        if unique_values <= 10 or df[label_column].dtype == 'bool':
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION

    def _get_hyperparameters(self, task_type: TaskType, user_input) -> Dict:
        """Get hyperparameters suggestion from LLM."""
        messages = [
            SystemMessage(content=Determine_model),
            HumanMessage(content=json.dumps({
                "task_type": task_type,
                "user_input": user_input
            }))
        ]
        
        response = self.llm.invoke(messages)
        print('def _get_hyperparameters:', json.loads(response.content))
        return json.loads(response.content)

    def _parse_user_input(self, user_input: str) -> Dict: # 提取用户输入，sequence 或者pdb文件
        """Parse user input to extract requirements."""
        messages = [
            SystemMessage(content=Parse_user_input2),
            HumanMessage(content=user_input)
        ]
        # print(Parse_user_input2[:10])
        
        response = self.llm.invoke(messages)
        # print(response)
        return json.loads(response.content)

    def process_input(self, user_input: str, file_path: str) -> ModelConfig:
        """Process user input and prepare model configuration."""
        # Read data file
        df = self._read_file(file_path)
        
        # Parse user requirements
        requirements = self._parse_user_input(user_input)
        print(requirements)

        task_type = requirements.get('task_type')
        pdb_ids = requirements.get('pdb_ids', [])
        sequences = requirements.get('sequences', [])
        
        # Analyze data structure
        data_analysis = self._analyze_data_structure(user_input, df)
        print('data_analysis:', data_analysis)
        
        # Determine task type and columns
        label_column = data_analysis['label_column']
        data_columns = data_analysis['data_columns']
        # task_type = self._determine_task_type(df, label_column)
        # print('task_type:', task_type)
        
        # Get hyperparameters suggestion
        hyperparameters = self._get_hyperparameters(task_type, user_input)
        
        # Create model configuration
        self.current_config = {
            'task_type': task_type,
            'file_path': file_path,
            'model_name': hyperparameters.get('recommended_model', 'esm2_t33_650M_UR50D'),
            'data_columns':data_columns,
            'label_column':label_column,
            # 'hyperparameters':hyperparameters,
            'file_path':file_path,
            'pdb_ids': pdb_ids,
            'sequences': sequences,
            'num_samples': hyperparameters.get('num_samples', 10),
            'num_epochs': hyperparameters.get('num_epochs', 10),
            'batch_size': hyperparameters.get('batch_size', 32),
            'lr': hyperparameters.get('lr'),
            'dropout': hyperparameters.get('dropout'),
            'accumulation_steps': hyperparameters.get('accumulation_steps'),
        }

        return self.current_config

    def run_automl(self) -> Dict:
        """Run AutoML based on current configuration."""
        if not self.current_config:
            raise ValueError("No configuration available. Run process_input first.")
        
        if self.current_config.task_type == TaskType.CLASSIFICATION:
            results = classification_main(self.current_config)
        else:
            results = regression_main(self.current_config)
        
        # Summarize results
        messages = [
            SystemMessage(content=Summary_output),
            HumanMessage(content=json.dumps(results))
        ]
        
        summary = self.llm.invoke(messages)
        return json.loads(summary.content)

def main():
    # Configuration
    config = {
        'inference_server_url': "https://ark.cn-beijing.volces.com/api/v3",
        'api_key': "72c991d1-8078-4c49-8b5e-7b33c4e26e04",
        'model_name': "ep-20240722030254-mrqhx"
    }
    print("AutoPE Configuration:")
    
    # Initialize AutoPE
    auto_pe = AutoPE(**config)
    
    print("Welcome to AutoPE! Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                break
                
            # Get file path from user or use default
            file_path = input("Enter file path (or press enter for default): ")
            if not file_path:
                file_path = '/home/netzone22/lftp/yungeng/ray/突变库数据1.03_with_mutated_sequences.xlsx'
                
            # Process input and get configuration
            model_config = auto_pe.process_input(user_input, file_path)
            print("\nTask Configuration:")
            print(f"Task Type: {model_config.task_type}")
            print(f"Selected Model: {model_config.model_name}")
            print(f"Data Columns: {model_config.data_columns}")
            print(f"Label Column: {model_config.label_column}")
            print("\nHyperparameters:")
            print(json.dumps(model_config.hyperparameters, indent=2))
            
            # Confirm with user
            proceed = input("\nProceed with AutoML? (y/n): ")
            if proceed.lower() == 'y':
                results = auto_pe.run_automl()
                print("\nAutoML Results:")
                print(json.dumps(results, indent=2))
            
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

if __name__ == "__main__":
    main()