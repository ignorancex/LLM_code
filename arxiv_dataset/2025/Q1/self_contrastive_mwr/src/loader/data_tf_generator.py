from typing import Dict, Tuple, Optional

import tensorflow as tf

from loader.data_generator import (DataGenerator, DataGlobalGenerator,
                                   DataRegionalGenerator, DataLocalGenerator,
                                   DataJointGenerator, DataContrastiveGenerator)



class TensorFlowDataGenerator():
    
    @staticmethod
    def _prepare_generators(dg: DataGenerator, batch_size: int,
                            output_shapes: Tuple[Dict[str, tf.TensorShape]],
                            output_types: Tuple[Dict[str, tf.dtypes.DType]],
                            seed: Optional[int] = None) -> Tuple[tf.data.Dataset]:
        """
        Prepare the train, validation, and test generators.

        Args:
            dg (DataGenerator): The DataGenerator object.
            batch_size (int): The batch size for the generators.
            output_shapes (Tuple[Dict[str, tf.TensorShape]]): The output shapes for each feature in the generators.
            output_types (Tuple[Dict[str, tf.dtypes.DType]]): The output types for each feature in the generators.
            seed (int, optional): The seed value for shuffling the train generator. Defaults to None.

        Returns:
            Tuple[tf.data.Dataset]: A tuple containing the train, validation, and test generators, along with the DataGenerator object.
        """
        buffer_size = dg.max_data

        generator_type = dg.train_generator
        train_generator = tf.data.Dataset.from_generator(generator_type,
                                                         output_types=output_types,
                                                         output_shapes=output_shapes)
        train_generator = train_generator.shuffle(buffer_size=buffer_size,
                                                  seed=seed,
                                                  reshuffle_each_iteration=True
                                                  ).batch(batch_size).prefetch(40)
        
        generator_type = dg.validation_generator
        validation_generator = tf.data.Dataset.from_generator(generator_type,
                                                              output_types=output_types,
                                                              output_shapes=output_shapes)
        validation_generator = validation_generator.batch(batch_size)
        
        generator_type = dg.test_generator
        test_generator = tf.data.Dataset.from_generator(generator_type,
                                                        output_types=output_types,
                                                        output_shapes=output_shapes)
        test_generator = test_generator.batch(batch_size)
        
        return train_generator, validation_generator, test_generator, dg


    @staticmethod
    def get_generators(data_path, train_ratio: float = 0.6, validation_ratio: float = 0.2,
                       test_ratio: float = 0.2, batch_size: int = 1,
                       contrastive: bool = False, train_percentage: Optional[float] = None,
                       seed: Optional[int] = None) -> Tuple[tf.data.Dataset]:  
        """
        Create TensorFlow data generators for training, validation, and testing for the base model.
        
        Args:
            data_path (str): The path to the data file.
            train_ratio (float, optional): The ratio of the data to use for training. Defaults to 0.6.
            validation_ratio (float, optional): The ratio of the data to use for validation. Defaults to 0.2.
            test_ratio (float, optional): The ratio of the data to use for testing. Defaults to 0.2.
            batch_size (int, optional): The batch size for the generators. Defaults to 1.
            contrastive (bool, optional): Whether to use contrastive learning. Defaults to False.
            train_percentage (float, optional): The percentage of the training data to use. Defaults to None.
            seed (int, optional): The seed value for shuffling the train generator. Defaults to None.

        Returns:
            Tuple[tf.data.Dataset]: A tuple containing the train, validation, and test generators.
        """    
        dg = DataGenerator(data_path,
                           train_ratio=train_ratio,
                           validation_ratio=validation_ratio,
                           test_ratio=test_ratio,
                           contrastive=contrastive,
                           train_subset=train_percentage,
                           seed=seed)
        
        output_shapes = ({'base_input': tf.TensorShape(dg.X_size)},
                         {'base_output': tf.TensorShape(dg.y_size)})
        
        output_types = ({'base_input': tf.float32},
                        {'base_output': tf.float32})
        
        if contrastive:
            output_shapes[1]['base_contrastive'] = tf.TensorShape(dg.y_size)
            output_types[1]['base_contrastive'] = tf.float32
        
        return TensorFlowDataGenerator._prepare_generators(dg, batch_size,
                                                           output_shapes,
                                                           output_types)
    
    

class TensorFlowDataGlobalGenerator(TensorFlowDataGenerator):

    @staticmethod
    def get_generators(data_path, train_ratio: float = 0.6, validation_ratio: float = 0.2,
                       test_ratio: float = 0.2, batch_size: int = 1,
                       contrastive: bool = False, train_percentage: Optional[float] = None,
                       seed: Optional[int] = None) -> Tuple[tf.data.Dataset]:
        """
        Create TensorFlow data generators for training, validation, and testing for the Global model.

        Args:
            data_path (str): The path to the data.
            train_ratio (float, optional): The ratio of training data. Defaults to 0.6.
            validation_ratio (float, optional): The ratio of validation data. Defaults to 0.2.
            test_ratio (float, optional): The ratio of testing data. Defaults to 0.2.
            batch_size (int, optional): The batch size. Defaults to 1.
            contrastive (bool, optional): Whether to generate contrastive data. Defaults to False.
            train_percentage (float, optional): The percentage of training data to use. Defaults to None.
            seed (int, optional): The random seed. Defaults to None.

        Returns:
            Tuple[tf.data.Dataset]: A tuple containing TensorFlow datasets for training, validation, and testing.
        """
        dg = DataGlobalGenerator(data_path,
                                 train_ratio=train_ratio,
                                 validation_ratio=validation_ratio,
                                 test_ratio=test_ratio,
                                 contrastive=contrastive,
                                 train_subset=train_percentage,
                                 seed=seed)
        
        output_shapes = ({'global_input': tf.TensorShape(dg.X_size),
                          'global_input_flipped': tf.TensorShape(dg.X_size)},
                         {'global_output': tf.TensorShape(dg.y_size)})
        
        output_types = ({'global_input': tf.float32,
                         'global_input_flipped': tf.float32},
                        {'global_output': tf.float32})

        if contrastive:
            output_shapes[1]['global_contrastive'] = tf.TensorShape(dg.y_size)
            output_types[1]['global_contrastive'] = tf.float32

        return TensorFlowDataGenerator._prepare_generators(dg, batch_size,
                                                           output_shapes,
                                                           output_types)
    
    
    
class TensorFlowDataRegionalGenerator(TensorFlowDataGenerator):

    @staticmethod
    def get_generators(data_path, train_ratio: float = 0.6, validation_ratio: float = 0.2,
                       test_ratio: float = 0.2, batch_size: int = 1,
                       contrastive: bool = False, train_percentage: Optional[float] = None,
                       seed: Optional[int] = None) -> Tuple[tf.data.Dataset]:   
        """
        Create TensorFlow data generators for training, validation, and testing for the Regional model.

        Args:
            data_path (str): The path to the data.
            train_ratio (float, optional): The ratio of training data. Defaults to 0.6.
            validation_ratio (float, optional): The ratio of validation data. Defaults to 0.2.
            test_ratio (float, optional): The ratio of testing data. Defaults to 0.2.
            batch_size (int, optional): The batch size. Defaults to 1.
            contrastive (bool, optional): Whether to generate contrastive data. Defaults to False.
            train_percentage (float, optional): The percentage of training data to use. Defaults to None.
            seed (int, optional): The random seed. Defaults to None.

        Returns:
            Tuple[tf.data.Dataset]: A tuple containing TensorFlow datasets for training, validation, and testing.
        """   
        dg = DataRegionalGenerator(data_path,
                                   train_ratio=train_ratio,
                                   validation_ratio=validation_ratio,
                                   test_ratio=test_ratio,
                                   contrastive=contrastive,
                                   train_subset=train_percentage,
                                   seed=seed)
        
        output_shapes = ({'regional_input_l': tf.TensorShape(dg.X_size),
                          'regional_input_r': tf.TensorShape(dg.X_size)},
                         {'regional_output': tf.TensorShape(dg.y_size)})
        
        output_types = ({'regional_input_l': tf.float32,
                         'regional_input_r': tf.float32},
                        {'regional_output': tf.float32})

        if contrastive:
            output_shapes[1]['regional_contrastive'] = tf.TensorShape(dg.y_size)
            output_types[1]['regional_contrastive'] = tf.float32

        return TensorFlowDataGenerator._prepare_generators(dg, batch_size,
                                                           output_shapes,
                                                           output_types)
    
    
    
class TensorFlowDataLocalGenerator(TensorFlowDataGenerator):

    @staticmethod
    def get_generators(data_path, train_ratio: float = 0.6, validation_ratio: float = 0.2,
                       test_ratio: float = 0.2, batch_size: int = 1,
                       contrastive: bool = False, train_percentage: Optional[float] = None,
                       seed: Optional[int] = None) -> Tuple[tf.data.Dataset]:
        """
        Create TensorFlow data generators for training, validation, and testing for the Local model.

        Args:
            data_path (str): The path to the data.
            train_ratio (float, optional): The ratio of training data. Defaults to 0.6.
            validation_ratio (float, optional): The ratio of validation data. Defaults to 0.2.
            test_ratio (float, optional): The ratio of testing data. Defaults to 0.2.
            batch_size (int, optional): The batch size. Defaults to 1.
            contrastive (bool, optional): Whether to generate contrastive data. Defaults to False.
            train_percentage (float, optional): The percentage of training data to use. Defaults to None.
            seed (int, optional): The random seed. Defaults to None.

        Returns:
            Tuple[tf.data.Dataset]: A tuple containing TensorFlow datasets for training, validation, and testing.
        """    
        dg = DataLocalGenerator(data_path,
                                train_ratio=train_ratio,
                                validation_ratio=validation_ratio,
                                test_ratio=test_ratio,
                                contrastive=contrastive,
                                train_subset=train_percentage,
                                seed=seed)
        
        shapes_dict = {}
        types_dict = {}
        for i in range(dg.X_input_size):
            key = f'local_input_{i}'
            shapes_dict[key] = tf.TensorShape(dg.X_size)
            types_dict[key] = tf.float32
        
        output_shapes = (shapes_dict,
                         {'local_output': tf.TensorShape(dg.y_size)})
        
        output_types = (types_dict,
                        {'local_output': tf.float32})

        if contrastive:
            output_shapes[1]['local_contrastive'] = tf.TensorShape(dg.y_size)
            output_types[1]['local_contrastive'] = tf.float32

        return TensorFlowDataGenerator._prepare_generators(dg, batch_size,
                                                           output_shapes,
                                                           output_types)
    
    
class TensorFlowDataJointGenerator(TensorFlowDataGenerator):

    @staticmethod
    def get_generators(data_path, train_ratio: float = 0.6, validation_ratio: float = 0.2,
                       test_ratio: float = 0.2, batch_size: int = 1,
                       contrastive: bool = False, train_percentage: Optional[float] = None,
                       seed: Optional[int] = None) -> Tuple[tf.data.Dataset]:
        """
        Create TensorFlow data generators for training, validation, and testing for the Joint model.

        Args:
            data_path (str): The path to the data.
            train_ratio (float, optional): The ratio of training data. Defaults to 0.6.
            validation_ratio (float, optional): The ratio of validation data. Defaults to 0.2.
            test_ratio (float, optional): The ratio of testing data. Defaults to 0.2.
            batch_size (int, optional): The batch size. Defaults to 1.
            contrastive (bool, optional): Whether to generate contrastive data. Defaults to False.
            train_percentage (float, optional): The percentage of training data to use. Defaults to None.
            seed (int, optional): The random seed. Defaults to None.

        Returns:
            Tuple[tf.data.Dataset]: A tuple containing TensorFlow datasets for training, validation, and testing.
        """     
        dg = DataJointGenerator(data_path,
                                train_ratio=train_ratio,
                                validation_ratio=validation_ratio,
                                test_ratio=test_ratio,
                                contrastive=contrastive,
                                train_subset=train_percentage,
                                seed=seed)
        
        shapes_dict = {}
        types_dict = {}
        for i in range(dg.X_local_input_size):
            key = f'joint_local_input_{i}'
            shapes_dict[key] = tf.TensorShape(dg.X_sizes['local'])
            types_dict[key] = tf.float32
        
        for key in ['joint_regional_input_l', 'joint_regional_input_r']:
            shapes_dict[key] = tf.TensorShape(dg.X_sizes['regional'])
            types_dict[key] = tf.float32
    
        for key in ['joint_global_input', 'joint_global_input_flipped']:
            shapes_dict[key] = tf.TensorShape(dg.X_sizes['global'])
            types_dict[key] = tf.float32
            
        output_shapes = (shapes_dict,
                         {'local_output': tf.TensorShape(dg.y_size),
                          'regional_output': tf.TensorShape(dg.y_size),
                          'global_output': tf.TensorShape(dg.y_size),
                          'joint_output': tf.TensorShape(dg.y_size)})
        
        output_types = (types_dict,
                        {'local_output': tf.float32,
                         'regional_output': tf.float32,
                         'global_output': tf.float32,
                         'joint_output': tf.float32})
        
        if contrastive:
            output_shapes[1]['joint_local_contrastive'] = tf.TensorShape(dg.y_size)
            output_shapes[1]['joint_regional_contrastive'] = tf.TensorShape(dg.y_size)
            output_shapes[1]['joint_global_contrastive'] = tf.TensorShape(dg.y_size)
            output_types[1]['joint_local_contrastive'] = tf.float32
            output_types[1]['joint_regional_contrastive'] = tf.float32
            output_types[1]['joint_global_contrastive'] = tf.float32
            

        return TensorFlowDataGenerator._prepare_generators(dg, batch_size,
                                                           output_shapes,
                                                           output_types)
    


class TensorFlowDataContrastiveGenerator(TensorFlowDataGenerator):

    @staticmethod
    def get_generators(data_path, train_ratio: float = 0.6, validation_ratio: float = 0.2,
                       test_ratio: float = 0.2, batch_size: int = 1,
                       train_percentage: Optional[float] = None,
                       seed: Optional[int] = None) -> Tuple[tf.data.Dataset]:
        """
        Create TensorFlow data generators for training, validation, and testing for the contrastive model.

        Args:
            data_path (str): The path to the data.
            train_ratio (float, optional): The ratio of training data. Defaults to 0.6.
            validation_ratio (float, optional): The ratio of validation data. Defaults to 0.2.
            test_ratio (float, optional): The ratio of testing data. Defaults to 0.2.
            batch_size (int, optional): The batch size. Defaults to 1.
            contrastive (bool, optional): Whether to generate contrastive data. Defaults to False.
            train_percentage (float, optional): The percentage of training data to use. Defaults to None.
            seed (int, optional): The random seed. Defaults to None.

        Returns:
            Tuple[tf.data.Dataset]: A tuple containing TensorFlow datasets for training, validation, and testing.
        """     
        dg = DataContrastiveGenerator(data_path,
                                      train_ratio=train_ratio,
                                      validation_ratio=validation_ratio,
                                      test_ratio=test_ratio,
                                      train_subset=train_percentage,
                                      seed=seed)
        
        output_shapes = ({'contrastive_input': tf.TensorShape(dg.X_size)},
                         {'contrastive_loss': tf.TensorShape(dg.y_size),
                          'contrastive_output': tf.TensorShape(dg.y_size)})
        
        output_types = ({'contrastive_input': tf.float32},
                        {'contrastive_loss': tf.float32,
                         'contrastive_output': tf.float32})

        return TensorFlowDataGenerator._prepare_generators(dg, batch_size,
                                                           output_shapes,
                                                           output_types)
    