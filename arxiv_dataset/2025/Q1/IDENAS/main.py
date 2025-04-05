# main.py
"""
This is the main script for running iterations on the InputModel.
"""

from input_model import InputModel

if __name__ == "__main__":
    model = InputModel("MNIST", "mobilenetv2", "Unet", 32, 1, 8)
    model.create_log_file()
    model.load_data()
    model.calculate_steps()
    model.preprocess_input()
    model.run_iterations()
