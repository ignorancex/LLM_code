import configparser
    
class ConfigManager:

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('/home/angel/config.ini')

    def get_main_path(self):
        return self.config['MAIN']['PATH']

    def get_clada_dictionary(self):
        return self.get_main_path() + '/' + self.config['DATA']['CLADA_DICTIONARY']
    
    def get_train_dataset(self):
        return self.get_main_path() + '/' + self.config['DATA']['TRAIN']

    def get_test_dataset(self):
        return self.get_main_path() + '/' + self.config['DATA']['TEST']

    def get_synthetic_dataset(self):
        return self.get_main_path() + '/' + self.config['DATA']['SYNTHETIC']

    def get_enable_synthetic(self):
        return self.config['DATA']['ENABLE_SYNTHETIC'].lower() == 'true'

    def get_error_detection_tokenizer(self):
        return self.get_main_path() + '/' + self.config['DETECTOR']['TOKENIZER']
    
    def get_error_detection_model(self):
        return self.get_main_path() + '/' + self.config['DETECTOR']['MODEL']

    def get_error_corrector_model(self):
        return self.get_main_path() + '/' + self.config['CORRECTOR']['MODEL']

    # Additional properties
    def get_converter_ivanchevski_resources_dir(self):
        return self.get_main_path() + '/' + self.config['CONVERTER']['IVANCHEVSKI_RESOURCES']

    def get_converter_drinovski_resources_dir(self):
        return self.get_main_path() + '/' + self.config['CONVERTER']['DRINOVSKI_RESOURCES']

    def get_stanza_dir(self):
        return self.get_main_path() + '/' + self.config['CONVERTER']['STANZA_DIR']