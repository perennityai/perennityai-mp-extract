
import gc
import json
import os
import glob
import time
import pandas as pd

import numpy as np
import tensorflow as tf
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Union, Tuple, List, Optional
from perennityai_feature_engine.utils import IndiciesSpecification
from perennityai_feature_engine.featureengineering import LandmarkFeatureExtractor
from perennityai_feature_engine.utils import CSVHandler
from perennityai_feature_engine.utils import Log

import configparser
config = configparser.ConfigParser()
config.read('features_config.ini')


VERBOSE = config['tf_constants']['VERBOSE']
SOURCE_MAX_LEN = config.getint('tf_constants', 'SOURCE_MAX_LEN')

PROCESS_FINGER_SPELLING_TF = config.getint('tf_constants', 'PROCESS_FINGER_SPELLING_TF')
PROCESS_KAGGLE_FINGER_SPELLING_TF = config.getint('tf_constants', 'PROCESS_KAGGLE_FINGER_SPELLING_TF')
PROCESS_GYNAMIC_GUESTURE_TF = config.getint('tf_constants','PROCESS_GYNAMIC_GUESTURE_TF')
PROCESS_STATIC_GESTURE_TF = config.getint('tf_constants', 'PROCESS_STATIC_GESTURE_TF')
PRO_FEATURE_EXTRACTION = config.getint('tf_constants', 'PRO_FEATURE_EXTRACTION')


"""
Why Remove Rows with NaN Values for Both Hands
1.  Lack of Useful Information: Rows with NaNs for both hands usually indicate that no relevant hand data was captured in those frames. 
    Such rows do not provide information about gestures and may introduce noise.
2.  Data Quality: Removing rows with all-NaN values for both hands can improve the quality and consistency of the dataset, 
    making it easier for models to learn from valid gesture data.
3.  Model Training Efficiency: Retaining rows with meaningful data helps focus the model on learning gesture patterns, potentially 
    improving performance and reducing training time.
"""


class FeatureProcessor:
    """
    A class to process gesture features and manage CSV and Parquet files for machine learning.

    Attributes:
        input_dir (str): Directory path where input data (train_landmarks folder and train.csv) is located.
        output_dir (str): Directory path where processed features and logs will be saved.
        processor (str) : Gesture processor
        extract (bool) : Gesture feature processing in process_dataset()
        overwrite (bool) : Overwrite generated tfrecord files
        runtime_name (str): Name based on the active processing type (e.g., 'dynamic', 'fingerspelling', 'static') for runtime-specific file naming.
        logger (Log): Logger instance for logging messages to a file, using the specified runtime name.
        parquet_path (str): Path to the train_landmarks directory containing Parquet files.
        features_path (str): Path to the directory where processed feature files will be saved.
        tfrecord_path (str): Path to the directory where TFRecord files will be saved.
        csv (CSVHandler): Instance of CSVHandler to handle CSV reading and writing.

    Files in `input_dir`:
        - train_landmarks folder: Contains Parquet files with columns `sequence_id` and 1629 points.
        - train.csv: A file with columns including `sequence_id`, `file_id`, `path`, `phrase`, and `participant_id`.
            - `file_id`: Matches the Parquet filename (with `.parquet` extension).
            - `sequence_id`: Unique identifier for each sample.
            - Other releases may also contain columns for `phrase`, `gender`, and `context`.

    Methods:
        __init__(self, csv_handler, input_dir, output_dir, processor): Initializes the class with paths and configurations for processing features.
    """

    def __init__(
        self, 
        processor: str,               # Required parameter
        input_dir: Optional[str] = '',     # Optional parameters with default values
        output_dir: Optional[str] = '',
        extract: bool = False,
        average_out_face: bool = False,
        overwrite: bool = True,
        pid: int = 0
    ):
        self.csv = CSVHandler()
        self.processor = processor
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.average_out_face = average_out_face
        self.overwrite = overwrite
        
        self.runtime_name, subfolder = self.get_runtime_name(self.processor)
        file_ext = '_avg' if self.average_out_face else ''

        if self.output_dir:
            self.logger = Log(log_file=os.path.join(self.output_dir, subfolder, self.runtime_name + f"features{file_ext}.log"), verbose=VERBOSE)
        else:
            self.logger = Log(log_file=os.path.join(os.getcwd(), self.runtime_name + f"features{file_ext}.log"), verbose=VERBOSE)

        self.logger.info("self.runtime_name ", self.runtime_name, self.processor)

        # Output path configurations
        self.features_path = os.path.join(self.output_dir, subfolder, self.runtime_name + 'dist_tfrecords' + file_ext)
        self.tfrecord_path = os.path.join(self.output_dir, subfolder, self.runtime_name + 'tfrecords' + file_ext)
        self.dataset_file = os.path.join(self.output_dir, subfolder, "federation_{pid}_{subfolder}.csv")
        self.parameters_json_filepath = os.path.join(self.output_dir, subfolder, self.runtime_name + f"processing_parameter{file_ext}.json")

        # Final output path of this stage
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.features_path, exist_ok=True)
            os.makedirs(self.tfrecord_path, exist_ok=True)

        # Initalize processors         
        self.__indicies_spec = IndiciesSpecification(logger=self.logger)
        self.__lm_feature_pro = LandmarkFeatureExtractor(self.__indicies_spec, 
                                                         average_out_face=self.average_out_face,
                                                         logger=self.logger)
        
        # Give command line parameters priority over config file
        self.process_feature_extraction = extract if extract else bool(PRO_FEATURE_EXTRACTION)
        
        self.logger.info("Initialized FeatureProcessor!")

    def validate_single_active_flag(*flags) -> bool:
        """
        Validates that no more than one flag is active (set to True) from the provided flags.

        Args:
            *flags: Boolean flags representing processor configurations.
                Examples:
                    PROCESS_FINGER_SPELLING_TF
                    PROCESS_KAGGLE_FINGER_SPELLING_TF
                    PROCESS_DYNAMIC_GESTURE_TF
                    PROCESS_STATIC_GESTURE_TF

        Returns:
            bool: True if zero or one flag is active, indicating valid configuration.

        Raises:
            ValueError: If more than one flag is set to True, indicating conflicting configurations.
        """
        active_flags = sum(flags)
        if active_flags > 1:
            raise ValueError("Only one processor flag can be active at a time.")
        return True

    def get_runtime_name(self, processor) -> str:
        """
        Determines the runtime processing name based on the active processing flags.

        This function checks which processing type is activated (dynamic, fingerspelling, or static) and 
        returns a corresponding name string. The returned name is used to label the processing runtime 
        appropriately.

        Returns:
            str: A string indicating the processing type. Possible values:
                - 'dynamic_': if dynamic gesture processing is enabled.
                - 'fingerspelling_': if fingerspelling gesture processing is enabled.
                - 'static_': if static gesture processing is enabled.

        Raises:
            ValueError: If none of the expected processing flags are enabled, indicating an unknown runtime name.
        """
        # Updated logic to set `processor_name`:
        self.processor_name = ''
        
        if processor:
            self.processor_name = processor.strip()
            subfolder = 'non_fingerspelling'
            if 'ingerspelling' in self.processor_name:
                subfolder = 'fingerspelling'
        elif self.validate_single_active_flag(
            PROCESS_FINGER_SPELLING_TF,
            PROCESS_KAGGLE_FINGER_SPELLING_TF,
            PROCESS_GYNAMIC_GUESTURE_TF,
            PROCESS_STATIC_GESTURE_TF
        ):
            subfolder = ''
            # Select processor based on active flag
            if PROCESS_FINGER_SPELLING_TF:
                self.processor_name = "fingerspelling"
                subfolder = 'fingerspelling'
            elif PROCESS_KAGGLE_FINGER_SPELLING_TF:
                self.processor_name = "kaggle_fingerspelling"
                subfolder = 'fingerspelling'
            elif PROCESS_GYNAMIC_GUESTURE_TF:
                self.processor_name = "dynamic"
                subfolder = 'non_fingerspelling'
            elif PROCESS_STATIC_GESTURE_TF:
                self.processor_name = "static"
                subfolder = 'non_fingerspelling'
        else:
            raise ValueError("Processor must be provided via config.ini or cmd argument")

        return self.processor_name + '_', subfolder
    
    def process_dataset(self, padding: bool = False, extract_only: bool = False) -> Optional[None]:
        """
        Processes dynamic, static, and finger-spelling gestures by reading train.csv or a Parquet file,
        then applying processors based on configuration in features_config.ini.

        This function is designed to handle the following:
            - Reads input from either a CSV or Parquet file, processing each sample.
            - Applies different processors based on activation flags in the configuration file.
            - Supports only one file type at a time (either CSV or Parquet).
            - Processes can include both Parquet-to-TFRecord conversion and feature extraction simultaneously.
            - Utilizes Euclidean distance comparison between points for all processors, except 'kaggle-fingerspelling',
            which only normalizes data, as each frame is expected to correspond to a single alphabetic letter.

        Args:
            pad_and_resize (bool): If True, applies padding and resizing to each sample during processing. Default is False.

        Returns:
            None: This function does not return any value; it performs processing and raises a ValueError for missing input.

        Raises:
            ValueError: If `self.input_dir` is not provided, indicating the input directory for batch processing is missing.
        """
        
        parameters = {}

        # Record the start time
        start_time = time.time()
        parameters['start_time'] = start_time
        parquet_path = os.path.join(self.input_dir, "train_landmarks")
        
        if os.path.exists(parquet_path):
            # Check parquet files
            if not os.listdir(parquet_path):
                raise ValueError("No .parquet filef found in ./train_landmarks folder") 
            
            # Define train file
            train_file = os.path.join(self.input_dir, "train.csv")
            # Ensure train file is provided
            if not os.path.exists(train_file):
                raise ValueError("train.csv file is missing in input_dir. Must have it and /train_landmarks to run!") 

            # Read
            dataset_df = self.csv.read_csv_file(train_file)

            if dataset_df.empty:
                self.logger.error("Train file cannot be empty!")
                return 
            
            parameters['num_samples'] = dataset_df.shape[0]

            pad_and_resize = padding
            parameters['padding'] = padding

            # Data processing development only. Todo: remove
            if 'batch_file_id' in dataset_df.columns:
                dataset_df['file_id'] = dataset_df['batch_file_id']

            if self.processor_name == 'dynamic':
                # Get dynamic columns
                extracted_feature_columns = []
                if self.process_feature_extraction:
                    extracted_feature_columns = self.__indicies_spec.get_dynamic_columns(average_out_face=self.average_out_face)

                # Get features indicies
                selected_feature_columns, _, _, _, _ = self.get_processor_indices(processor='dynamic')

            elif self.processor_name == 'fingerspelling' or self.processor_name == 'kaggle_fingerspelling':
                selected_feature_columns, RHAND_IDX, LHAND_IDX, _, _ = self.__indicies_spec.process_fingerspelling_idx()
                extracted_feature_columns = [] # 78 columns
                if self.process_feature_extraction and self.processor_name == 'fingerspelling':
                    extracted_feature_columns = self.__indicies_spec.get_fingerspelling_columns()
                else:
                    extracted_feature_columns = self.__indicies_spec.get_fingerspelling_columns(kaggle=True)
                    
            elif self.processor_name == 'static':
                selected_feature_columns, RHAND_IDX, LHAND_IDX = self.__indicies_spec.process_static_idx()
                
                extracted_feature_columns = [] # i don't know yet
                if self.process_feature_extraction:
                    extracted_feature_columns = self.__indicies_spec.get_static_columns()
                    pass

            parameters['selected_feature_columns'] = selected_feature_columns
            parameters['extracted_feature_columns'] = extracted_feature_columns

            # Loop through each file_id
            for file_id in dataset_df.file_id.unique():
                file_id = int(file_id)
                print()
                self.logger.info('Processing : ', file_id)

                # Filter train.csv and fetch entries only for the relevant file_id
                file_df = dataset_df.loc[dataset_df["file_id"] == file_id]
                #self.logger.debug("file_df shape : ", file_df.shape)

                if file_df.empty:
                    raise ValueError(f"Error : {file_id}.parquet is empty! ")
                
                # Parquet file name
                pq_file = os.path.join(parquet_path, f"{file_id}.parquet")
                
                # Fetch the parquet file
                parquet_df = self.csv.read_parquet_file(pq_file, columns=['sequence_id'] + selected_feature_columns)

                #self.logger.debug("parquet_df : ", list(parquet_df.columns[0:10]))
                self.logger.info("parquet_df shape: ", parquet_df.shape)
                #self.logger.debug("parquet_df seq_id: ", list(parquet_df.sequence_id.unique()[0:10]))

                # Compare sequence ids
                sed_ids_t = file_df['sequence_id'].unique()
                sed_ids_p = parquet_df[parquet_df['sequence_id'].isin(sed_ids_t)]['sequence_id'].unique()

                diff = set(sed_ids_t) - set(sed_ids_p)
                
                if diff:
                    self.logger.critical("sed_ids_t : ", len(sed_ids_t), sed_ids_t[0:25])
                    self.logger.critical("sed_ids_p : ", len(sed_ids_p), sed_ids_p[0:25])
                    self.logger.critical("diff : ", list(diff)[0:25])
                    raise ValueError(f"missing sequence_id : {list(diff)[0:25]}")
                                    

                # Remove rows where all values in both hands are NaN                 
                parquet_df = parquet_df.dropna(subset=self.__indicies_spec.get_hand_columns(), how='all')

                parquet_df['sequence_id'] = parquet_df['sequence_id'].astype(int)
                parquet_seq_ids = parquet_df['sequence_id'].unique().tolist()

                if not parquet_df.empty:
                    #########################################################################################
                    # Process Feature Extraction
                    #########################################################################################
                    if self.process_feature_extraction:
                        tf_file = os.path.join(self.features_path, f"{file_id}.tfrecord")
                        if not os.path.exists(tf_file) or self.overwrite:
                            # Initialize the pointer to write the output of 
                            # each `for loop` below as a sequence into the file.
                            with tf.io.TFRecordWriter(tf_file) as file_writer:
                                # Loop through each sequence in file.
                                for seq_id, phrase in zip(file_df.sequence_id, file_df.phrase):
                                    # Fetch sequence data
                                    if int(seq_id) in parquet_seq_ids:
                                        # Initialize variable
                                        tf_feature = tf.constant([])
                                        frames_df = parquet_df[parquet_df['sequence_id'] == int(seq_id)] # MediaPipe Landmarks
                                        # Convert to numpy
                                        frames_np = frames_df.to_numpy()
                                        # Convert DataFrame to Tensor
                                        frame_tf = tf.convert_to_tensor(frames_df.to_numpy(), dtype=tf.float32)

                                        #self.logger.debug("frame_tf : ", frame_tf.shape)

                                        # Initializer write detector
                                        is_writable = False

                                        # Perform feature extraction
                                        if self.processor_name == 'dynamic':
                                            # Calculate the Euclidean distance between each points and reference points. 
                                            tf_feature = self.extract_frame_features(frame_tf, processor='dynamic', padding=padding)

                                            #self.logger.debug("tf_feature_w_seq ", tf_feature.shape)
                                            #self.logger.debug("extracted_feature_columns : ", len(extracted_feature_columns))
                                            # Convert feature to numpy
                                            frames_np = tf_feature.numpy()

                                            #self.logger.debug("tf_feature: ", tf_feature.shape, " frames_np: ", frames_np.shape, " extracted_feature_columns :", len(extracted_feature_columns))
                                            # Activate writer
                                            is_writable = True
                                        elif self.processor_name == 'kaggle_fingerspelling': # Use fingerspelling column # Todo
                                            # Calculate the Euclidean distance between each points and reference points. 
                                            tf_feature = self.extract_frame_features(frame_tf, processor='kaggle_fingerspelling', padding=padding)
                                            # Convert feature to numpy
                                            frames_np = tf_feature.numpy()
                                            # Activate writer
                                            is_writable = True
                                        elif self.processor_name == 'fingerspelling':
                                            # Calculate the Euclidean distance between each points and reference points. 
                                            tf_feature = self.extract_frame_features(frame_tf, processor='fingerspelling', padding=padding)
                                            #self.logger.debug('tf_feature ', tf_feature.shape)
                                            # Convert feature to numpy
                                            frames_np = tf_feature.numpy()
                                            # Activate writer
                                            is_writable = True
                                        elif self.processor_name == 'static':
                                            # Use static columns column # Todo. Two hands only
                                            # Calculate the Euclidean distance between each points and reference points. 
                                            tf_feature = self.extract_frame_features(frame_tf, processor='static', padding=padding)
                                            # Convert feature to numpy
                                            frames_np = tf_feature.numpy()
                                            # Activate writer
                                            is_writable = True
                                        else:
                                            raise ValueError(f"Activate a type of process")
                                        
                                        if is_writable:
                                            # Write
                                            features = {extracted_feature_columns[i]: tf.train.Feature(
                                                    float_list=tf.train.FloatList(value=frames_np[:, i])) for i in range(len(extracted_feature_columns))}
                                            features["phrase"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(phrase, 'utf-8')]))
                                            record_bytes = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                                            file_writer.write(record_bytes)
                                            print('.', end='')

                                        if 'output_feature_shape' not in parameters:
                                            parameters['output_dist_feature_shape'] = tf_feature.shape.as_list()
                                            parameters['input_dist_feature_shape'] = frame_tf.shape.as_list()
                                        del frames_np
                                        del tf_feature

                                    else:
                                        raise ValueError(f"Missing Data: sequence_id: {seq_id}, phrase :{phrase}")
                        else:
                            self.logger.info(f"Extracted feature found {tf_file}, skipping!")

                    #########################################################################################
                    # Extract and write selected features
                    #########################################################################################
                    if not extract_only:
                        tf_file = os.path.join(self.tfrecord_path, f"{file_id}.tfrecord")
                        if not os.path.exists(tf_file) or self.overwrite:
                            # Write unprocessed file
                            with tf.io.TFRecordWriter(tf_file) as file_writer:
                                # Loop through each sequence in file.
                                for seq_id, phrase in zip(file_df.sequence_id, file_df.phrase):
                                    # Fetch sequence data
                                    if int(seq_id) in parquet_seq_ids:
                                        frames_df = parquet_df[parquet_df['sequence_id'] == int(seq_id)] # MediaPipe Landmarks
                                    
                                        # Assuming `frames_df` is a DataFrame and `seq_id` is the sequence ID value
                                        frames_np = frames_df.to_numpy()

                                        # Create a column of `seq_id` values with the same number of rows as `frames_np`
                                        seq_id_column = np.full((frames_np.shape[0], 1), seq_id, dtype=np.float32)

                                        # Concatenate `seq_id_column` with `frames_np` along the second axis (columns)
                                        frames_np = np.concatenate([seq_id_column, frames_np], axis=1)

                                        if self.processor_name == 'dynamic':
                                            # Process features
                                            features = {selected_feature_columns[i]: tf.train.Feature(
                                            float_list=tf.train.FloatList(value=frames_np[:, i])) for i in range(len(selected_feature_columns))}
                                            features["phrase"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(phrase, 'utf-8')]))
                                            record_bytes = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                                            file_writer.write(record_bytes)
                                            print('.', end='')
                                        elif self.processor_name == 'kaggle_fingerspelling': # Use fingerspelling column # Todo
                                            # Calculate the number of NaN values in each hand landmark
                                            r_nonan = np.sum(np.sum(np.isnan(frames_np[:, RHAND_IDX]), axis = 1) == 0)
                                            l_nonan = np.sum(np.sum(np.isnan(frames_np[:, LHAND_IDX]), axis = 1) == 0)
                                            no_nan = max(r_nonan, l_nonan)
                                            
                                            if 2 * len(phrase) < no_nan: # verify. maybe try longer length of nan
                                                features = {selected_feature_columns[i]: tf.train.Feature(
                                                    float_list=tf.train.FloatList(value=frames_np[:, i])) for i in range(len(selected_feature_columns))}
                                                features["phrase"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(phrase, 'utf-8')]))
                                                record_bytes = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                                                file_writer.write(record_bytes)
                                                print('.', end='')
                                            pass
                                        elif self.processor_name == 'fingerspelling':
                                            # Calculate the number of NaN values in each hand landmark
                                            r_nonan = np.sum(np.sum(np.isnan(frames_np[:, RHAND_IDX]), axis = 1) == 0)
                                            l_nonan = np.sum(np.sum(np.isnan(frames_np[:, LHAND_IDX]), axis = 1) == 0)
                                            no_nan = max(r_nonan, l_nonan)

                                            if 2 * len(phrase) < no_nan:
                                                features = {selected_feature_columns[i]: tf.train.Feature(
                                                    float_list=tf.train.FloatList(value=frames_np[:, i])) for i in range(len(selected_feature_columns))}
                                                features["phrase"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(phrase, 'utf-8')]))
                                                record_bytes = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                                                file_writer.write(record_bytes)
                                                print('.', end='')
                                        elif self.processor_name == 'static':
                                            features = {selected_feature_columns[i]: tf.train.Feature(
                                                    float_list=tf.train.FloatList(value=frames_np[:, i])) for i in range(len(selected_feature_columns))}
                                            features["phrase"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(phrase, 'utf-8')]))
                                            record_bytes = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                                            file_writer.write(record_bytes)
                                            print('.', end='')
                                        else:
                                            raise ValueError(f"Activate a type of process")
                                        
                                        if 'output_feature_shape' not in parameters:
                                            parameters['output_feature_shape'] = frames_np.shape
                                            parameters['input_feature_shape'] = frames_np.shape

                                        del frames_np
                                    else:
                                        raise ValueError(f"Missing Data: sequence_id: {seq_id}, phrase :{phrase}")
                        else:          
                            self.logger.info(f"Selected feature found {tf_file}, skipping!")
                       
                else:       
                    raise ValueError("parquet_df is empty!")
                
                # garbage collector
                del parquet_df
                del file_df
            self.logger.info(f"\nFinished processing {file_id}")
                
            del dataset_df 
        else:
            self.logger.info("Partquet not found : ", parquet_path)
        gc.collect()       
        # Calculate and print the runtime
        end_time = time.time()
        runtime = end_time - start_time

        # Load Parameters
        parameters['runtime'] = runtime
        parameters['PRO_FEATURE_EXTRACTION'] = PRO_FEATURE_EXTRACTION
        parameters['PROCESS_FINGER_SPELLING_TF'] = PROCESS_FINGER_SPELLING_TF
        parameters['PROCESS_STATIC_GESTURE_TF'] = PROCESS_STATIC_GESTURE_TF
        parameters['input_dir'] = self.input_dir
        parameters['output_dir'] = self.output_dir
        parameters['average_out_face'] = self.average_out_face
        parameters['extract_only'] = extract_only
        parameters['extract'] = self.process_feature_extraction


        # Save the paramters JSON to a file
        with open(self.parameters_json_filepath, 'w') as json_file:
            # Convert word index to JSON format
            parameters_json = json.dumps(parameters)
            # Write
            json_file.write(parameters_json)
            self.logger.info(f"Succesfully written : {self.parameters_json_filepath}")

        # Write Dataset file
        self.csv.write_csv_file(dataset_df, self.dataset_file)
        self.logger.info(f"Runtime: {runtime:.2f} seconds")

    def extract_frame_features(self, data: Union[str, pd.DataFrame, tf.Tensor, list], 
                                     processor: str = 'dynamic', 
                                     padding: bool = False, 
                                     average_out_face: Optional[bool] = None) -> tf.Tensor:
        """
        Extracts frame features by processing input data through different gesture processors.

        Args:
            data (Union[str, pd.DataFrame, tf.Tensor, list]): The input data to process (expecting sequence_id + 1629 points). Accepted types are:
                - str: File path to a CSV file containing the data.
                - pd.DataFrame: DataFrame containing landmark data.
                - tf.Tensor: Tensor containing the landmark data.
                - list: List of data points that will be converted to a Tensor.
            processor (str): Specifies the type of processor to use. Options are:
                - 'dynamic': Processes data for dynamic gestures.
                - 'fingerspelling': Processes data for fingerspelling gestures.
                - 'kaggle-fingerspelling': Processes data specifically for Kaggle fingerspelling gestures.
                - 'static': Processes data for static gestures.
                Default is 'dynamic'.
            padding (bool): If True, applies padding and resizing to the input data during processing. Default is False.
            average_out_face(bool): If True, applies mean calculation to all face distance feature. (e.g 128,3)

        Returns:
            tf.Tensor: A tensor containing the processed frame features.

        Raises:
            ValueError: If an unsupported `processor` type or `data` type is provided.
        """
        # Initialize an empty tensor
        x = tf.constant([])

       # Override if set
        this_average_out_face = average_out_face if average_out_face is not None else self.average_out_face

        # Convert data to tensor format if needed and assign processor data
        if isinstance(data, str):
            # Load data from CSV if `data` is a file path
            if os.path.isfile(data) and '.csv' in data:
                x = tf.convert_to_tensor(self.csv.read_csv_file(data).to_numpy(), dtype=tf.float32)
            else:
                raise ValueError(f"Invalid file path or file type provided: {data}")
        elif isinstance(data, pd.DataFrame):
            x = tf.convert_to_tensor(data.to_numpy(), dtype=tf.float32)
        elif isinstance(data, tf.Tensor):
            x = data
        elif isinstance(data, list):
            x = tf.convert_to_tensor(data, dtype=tf.float32)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Process the data based on the specified processor type
        if processor == 'dynamic':
            x = self.__lm_feature_pro.process_dynamic_gesture(x, pad_and_resize=padding, average_out_face=this_average_out_face)
        elif processor == 'fingerspelling':
            x = self.__lm_feature_pro.process_fingerspelling_gesture(x, pad_and_resize=padding)
        elif processor == 'kaggle_fingerspelling':
            x = self.__lm_feature_pro.process_kaggle_fingerspelling_gesture(x, pad_and_resize=padding)
        elif processor == 'static':
            x = self.__lm_feature_pro.process_static_gesture(x, pad_and_resize=padding)
        else:
            raise ValueError("Unsupported processor type provided.")

        return x

    def get_processor_indices(self, processor: str = 'dynamic') -> Tuple[List[int], ...]:
        """
        Retrieves index lists for specified processor types, defining the structure of landmark indices for each type.

        Args:
            processor (str): Specifies the processor type to use for retrieving indices. Options are:
                - 'dynamic': Retrieves indices for feature columns, right hand, left hand, pose, and face.
                - 'fingerspelling': Retrieves indices for feature columns, right hand, left hand, right pose, and left pose.
                - 'kaggle-fingerspelling': Similar to fingerspelling, retrieves indices for feature columns, right hand, left hand, right pose, and left pose.
                - 'static': Retrieves indices for feature columns, right hand, and left hand.

        Returns:
            Tuple[List[int], ...]: A tuple of lists, where each list contains indices corresponding to the selected processor type.
                - For 'dynamic', returns a 5-element tuple of lists: (feature_columns, RHAND_IDX, LHAND_IDX, POSE_IDX, FACE_IDX).
                - For 'fingerspelling' and 'kaggle-fingerspelling', returns a 5-element tuple of lists: (feature_columns, RHAND_IDX, LHAND_IDX, RPOSE_IDX, LPOSE_IDX).
                - For 'static', returns a 3-element tuple of lists: (feature_columns, RHAND_IDX, LHAND_IDX).

        Raises:
            ValueError: If an unsupported processor type is provided.
        """
        out = ()
        
        # Process the data based on the specified processor type
        if processor == 'dynamic':
            out = self.__indicies_spec.process_dynamic_idx()  # feature_columns, RHAND_IDX, LHAND_IDX, POSE_IDX, FACE_IDX
        elif processor == 'fingerspelling' or processor == 'kaggle_fingerspelling':
            out = self.__indicies_spec.process_fingerspelling_idx()  # feature_columns, RHAND_IDX, LHAND_IDX, RPOSE_IDX, LPOSE_IDX
        elif processor == 'static':
            out = self.__indicies_spec.process_static_idx()  # feature_columns, RHAND_IDX, LHAND_IDX
        else:
            raise ValueError("Unsupported processor type provided.")
        
        return out
    
    def resize_pad(self, x):
        return self.__lm_feature_pro.resize_pad(x)


