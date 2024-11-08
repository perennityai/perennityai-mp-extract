import numpy as np
from .feature_config_loader import FeatureConfigLoader


import configparser
config = configparser.ConfigParser()
config.read('features_config.ini')

ALL_FEATURE_COLUMNS = config['all_features']['ALL_FEATURE_COLUMNS']
ALL_FEATURE_COLUMNS = ALL_FEATURE_COLUMNS.split(" ") if not "\t" in ALL_FEATURE_COLUMNS else ALL_FEATURE_COLUMNS.split("\t")
AVERAGE_FACE_FEATURES = config['tf_constants']['AVERAGE_FACE_FEATURES']



loader = FeatureConfigLoader()
load_expression_landmark_indices = loader.load_expression_landmark_indices
load_body_pose_landmark_indices  = loader.load_body_pose_landmark_indices
load_hand_landmark_indices  = loader.load_hand_landmark_indices


class IndiciesSpecification:
    def __init__(self, logger=None):
        self.logger=logger
        self.hand_indices =  [val for subdict in load_hand_landmark_indices().values() for val in (subdict.values() if isinstance(subdict, dict) else [subdict]) for val in (val if isinstance(val, list) else val.values())]

    def get_hand_columns(self):
        """
        Generates column names for right and left hand features, including x, y, and z coordinates.

        Returns:
            list: A list of column names for hand features, structured as x, y, z coordinates for each hand.
        """
        return [
            f'{axis}_right_hand_{i}' for axis in ['x', 'y', 'z'] for i in self.hand_indices
        ] + [
            f'{axis}_left_hand_{i}' for axis in ['x', 'y', 'z'] for i in self.hand_indices
        ]

    def get_dynamic_columns(self, average_out_face = None):
        """
        Generates dynamic column names based on hand, pose, and face feature indices.
        
        Returns:
            list: A list of column names for the dynamic feature set, including hand, pose, and face features.
        """
        def get_dynamic_outgoing_features(pose=None, l_hand=None, r_hand=None, face=None):
            """
            Helper function to generate outgoing feature columns based on input indices.

            Args:
                pose (list): Indices for pose features.
                l_hand (list): Indices for left hand features.
                r_hand (list): Indices for right hand features.
                face (list): Indices for face features.

            Returns:
                list: A list of dynamically generated column names for each feature set.
            """
            if r_hand is not None and l_hand is not None and pose is not None and face is not None:
                # Generate columns based on feature order: hand, pose, face
                features_columns = (
                    [f'{axis}_right_hand_{i}' for axis in ['x', 'y', 'z'] for i in r_hand] +
                    [f'{axis}_left_hand_{i}' for axis in ['x', 'y', 'z'] for i in l_hand] +
                    [f'{axis}_pose_{i}' for axis in ['x', 'y', 'z'] for i in pose]
                )

                if AVERAGE_FACE_FEATURES:
                    # If averaging, use a single 'mean' face feature
                    features_columns += [f'{axis}_face_mean' for axis in ['x', 'y', 'z']]
                else:
                    # Otherwise, use detailed indices for each face feature
                    features_columns += [f'{axis}_face_{i}' for axis in ['x', 'y', 'z'] for i in face]
            else:
                raise ValueError("All feature sets (pose, left hand, right hand, face) must be provided.")

            return features_columns
        
        L_HAND = self.hand_indices  # Indices for left hand features
        R_HAND = self.hand_indices  # Indices for right hand features

        # Define indices for pose, left hand, right hand, and face based on the feature extraction requirements
        POSE = list(range(sum(len(parts) * (len(parts) - 1) // 2 for parts in load_body_pose_landmark_indices().values())))

        # Override average_out_face if specified, otherwise use class default
        this_average_out_face = average_out_face if average_out_face is not None else AVERAGE_FACE_FEATURES

        # Face indices depend on whether averaging is applied
        if this_average_out_face:
            face = ['mean']
        else:
            # Flatten all levels of nested lists
            face = list(load_expression_landmark_indices().keys()) # DBased on total face features

        # Generate the dynamic columns based on the defined indices
        columns = get_dynamic_outgoing_features(pose=POSE, l_hand=L_HAND, r_hand=R_HAND, face=face)

        # Log the details of generated columns
        print("get_dynamic_outgoing_features : ", len(columns))
        #self.logger.debug(f"{len(columns)} columns generated: pose ({len(POSE)}), hand ({len(L_HAND)})")

        return columns
    
    def get_fingerspelling_columns(self, kaggle: bool = False) -> list:
        """
        Generates a list of column names for fingerspelling feature extraction.

        Args:
            kaggle (bool): If True, includes pose columns for Kaggle data processing. Default is False.

        Returns:
            list: A list of column names including x, y, z coordinates for hand and optional pose landmarks.
        """
        # Generate hand columns dynamically based on `self.hand_indices`
        columns = [f'{axis}_hand_{i}' for axis in ['x', 'y', 'z'] for i in self.hand_indices]

        if kaggle:
            # For Kaggle data, add pose columns
            pose_indices = list(range(5))  # Assuming 5 pose indices
            columns += [f'{axis}_pose_{i}' for axis in ['x', 'y', 'z'] for i in pose_indices]

        return columns
    
    def get_static_columns(self):
        """
        Generates column names for static features, including x, y, and z coordinates for right and left hands.

        Returns:
            list: A list of column names for static features, structured as x, y, z coordinates for each hand.
        """
        # Generate columns dynamically for both right and left hands
        columns = [
            f'{axis}_right_hand_{i}' for axis in ['x', 'y', 'z'] for i in self.hand_indices
        ] + [
            f'{axis}_left_hand_{i}' for axis in ['x', 'y', 'z'] for i in self.hand_indices
        ]
        return columns

    
    def get_selected_features(self, pose=None, l_hand=None, r_hand=None, face=None):
        """
        Arg:
            pose:
            l_hand:
            r_hand:
            face:
            Note: if all arguments is none, then return all features.
        """
        X = []
        Y = []
        Z = []
        if r_hand is not None and l_hand is not None and pose is not None and face is not None: # dynamic
            X = [f'x_right_hand_{i}' for i in r_hand] + [f'x_left_hand_{i}' for i in l_hand] + [f'x_pose_{i}' for i in pose] + [f'x_face_{i}' for i in face]
            Y = [f'y_right_hand_{i}' for i in r_hand] + [f'y_left_hand_{i}' for i in l_hand] + [f'y_pose_{i}' for i in pose] + [f'y_face_{i}' for i in face]
            Z = [f'z_right_hand_{i}' for i in r_hand] + [f'z_left_hand_{i}' for i in l_hand] + [f'z_pose_{i}' for i in pose] + [f'z_face_{i}' for i in face]
        elif  r_hand is not None and l_hand is not None and pose is not None and face is None: # fingerspelling
            X = [f'x_right_hand_{i}' for i in r_hand] + [f'x_left_hand_{i}' for i in l_hand] + [f'x_pose_{i}' for i in pose]
            Y = [f'y_right_hand_{i}' for i in r_hand] + [f'y_left_hand_{i}' for i in l_hand] + [f'y_pose_{i}' for i in pose]
            Z = [f'z_right_hand_{i}' for i in r_hand] + [f'z_left_hand_{i}' for i in l_hand] + [f'z_pose_{i}' for i in pose]
        elif  r_hand is not None and l_hand is not None and pose is None and face is None: # static
            X = [f'x_right_hand_{i}' for i in r_hand] + [f'x_left_hand_{i}' for i in l_hand]
            Y = [f'y_right_hand_{i}' for i in r_hand] + [f'y_left_hand_{i}' for i in l_hand]
            Z = [f'z_right_hand_{i}' for i in r_hand] + [f'z_left_hand_{i}' for i in l_hand]
        elif r_hand is not None and l_hand is not None and face is not None: # potential dynamic
            X = [f'x_right_hand_{i}' for i in r_hand] + [f'x_left_hand_{i}' for i in l_hand]
            Y = [f'y_right_hand_{i}' for i in r_hand] + [f'y_left_hand_{i}' for i in l_hand]
            Z = [f'z_right_hand_{i}' for i in r_hand] + [f'z_left_hand_{i}' for i in l_hand]
        elif pose is None and l_hand is None and r_hand is None and face is None:
            raise ValueError("Provide atleast two hands parameters")

        features_columns = X + Y + Z
        return features_columns

    def process_dynamic_idx(self, face=None):
        # Face indices
        expression_landmark_indices = load_expression_landmark_indices()

        FACE = sorted([index for indices in expression_landmark_indices.values() for index in indices])

        body_pose_landmark_indices = load_body_pose_landmark_indices()

        POSE = sorted({index for region in body_pose_landmark_indices.values() for part in region.values() for index in part})

        L_HAND =  self.hand_indices
        R_HAND =  self.hand_indices

        #self.logger.debug("self.hand_indices : ", self.hand_indices)

        # Extract selected columns
        FEATURE_COLUMNS = self.get_selected_features(pose=POSE, l_hand=L_HAND, r_hand=R_HAND, face=FACE)

        #self.logger.debug("FEATURE_COLUMNS len : ", len(FEATURE_COLUMNS)) # ALL_FEATURE_COLUMNS
        #self.logger.debug("ALL_FEATURE_COLUMNS len : ", len(ALL_FEATURE_COLUMNS))

        # Extract indices of all features
        X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "x_" in col]
        Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "y_" in col]
        Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "z_" in col]
        
        # Extract indices of body part features
        # Hands
        RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "_right_hand_" in col]
        LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "_left_hand_" in col]

        # Face 
        FACE_IDX = [
            i for i, col in enumerate(FEATURE_COLUMNS) \
                if  "_face_" in col and int(col[-3:].replace("e_", "").replace("_", "")) in FACE]
        
        # Pose
        POSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "_pose_" in col and int(col[-2:].replace("_", "")) in POSE]


        if len(RHAND_IDX)//(len(RHAND_IDX)//3) % 3 != 0:
            raise ValueError("Error in RHAND_IDX dimension!")
        
        if len(LHAND_IDX)//(len(LHAND_IDX)//3) % 3 != 0:
            raise ValueError("Error in LHAND_IDX dimension!")
        
        if len(POSE_IDX)//(len(POSE_IDX)//3) % 3 != 0:
            raise ValueError("Error in POSE_IDX dimension!")
        
        if len(FACE_IDX)//(len(FACE_IDX)//3) % 3 != 0:
            raise ValueError("Error in FACE_IDX dimension!")

        if (len(RHAND_IDX) + len(LHAND_IDX) + len(FACE_IDX) + len(POSE_IDX)) != len(FEATURE_COLUMNS) and \
        (len(X_IDX) + len(Y_IDX) + len(Z_IDX)) != len(FEATURE_COLUMNS):
            raise ValueError('Error in extracting PROCESS_GYNAMIC_GUESTURE_TF indices!')
        
        
        return FEATURE_COLUMNS, RHAND_IDX, LHAND_IDX, POSE_IDX, FACE_IDX
    
    def get_finger_spelling_indices(self, kaggle: bool = False):
        """
        Retrieves index lists for fingerspelling feature extraction, specifying landmarks for right hand, left hand, right pose, and left pose.

        Args:
            kaggle (bool): If True, omits the initial pose indices (11 for left pose and 12 for right pose) 
                        commonly used in non-Kaggle data. Default is False.

        Returns:
            Tuple[List[int], List[int], List[int], List[int]]:
                - R_HAND (List[int]): Indices for right hand landmarks (0-20).
                - L_HAND (List[int]): Indices for left hand landmarks (0-20).
                - RPOSE (List[int]): Indices for right pose landmarks (specific indices based on `kaggle` parameter).
                - LPOSE (List[int]): Indices for left pose landmarks (specific indices based on `kaggle` parameter).
        """
        # Define default left and right pose indices for fingerspelling features
        LPOSE = [13, 15, 17, 19, 21]
        RPOSE = [14, 16, 18, 20, 22]

        # Modify pose indices for non-Kaggle data
        if not kaggle:
            LPOSE = [11] + LPOSE  # Add index 11 for left pose
            RPOSE = [12] + RPOSE  # Add index 12 for right pose

        # Define hand indices for right and left hands (0-20 for each hand)
        L_HAND = [i for i in range(21)]
        R_HAND = [i for i in range(21)]

        return R_HAND, L_HAND, RPOSE, LPOSE

    
    def process_fingerspelling_idx(self, kaggle=False):
        # Features
        
        R_HAND, L_HAND, RPOSE, LPOSE = self.get_finger_spelling_indices(kaggle=kaggle)
        POSE = LPOSE + RPOSE

        # Extract selected columns
        FEATURE_COLUMNS = self.get_selected_features(pose=POSE, l_hand=L_HAND, r_hand=R_HAND)
        num_features = len(FEATURE_COLUMNS)//3

        # Extract indices of all features
        X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "x_" in col]
        Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "y_" in col]
        Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "z_" in col]

        # Extract indices of body part features
        # Hands
        RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "_right_hand_" in col]
        LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "_left_hand_" in col]
        # Pose
        RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in RPOSE]
        LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in LPOSE]
        POSE_IDX = RPOSE_IDX + LPOSE_IDX

        
        if len(RHAND_IDX)//(len(RHAND_IDX)//3) % 3 != 0:
            raise ValueError("Error in RHAND_IDX dimension!")
        
        if len(LHAND_IDX)//(len(LHAND_IDX)//3) % 3 != 0:
            raise ValueError("Error in LHAND_IDX dimension!")
        
        if len(POSE_IDX)//(len(POSE_IDX)//3) % 3 != 0:
            raise ValueError("Error in POSE_IDX dimension!")
        

        if (len(RHAND_IDX) + len(LHAND_IDX) + len(RPOSE_IDX) + len(LPOSE_IDX)) != len(FEATURE_COLUMNS) and \
        (len(X_IDX) + len(Y_IDX) + len(Z_IDX)) != len(FEATURE_COLUMNS):
            raise ValueError('Error in extracting PROCESS_FINGER_SPELLING_TF indices!')
        
        return FEATURE_COLUMNS, RHAND_IDX, LHAND_IDX, RPOSE_IDX, LPOSE_IDX

    def process_static_idx(self):
        # Features
        LPOSE = [13, 15, 17, 19, 21]
        RPOSE = [14, 16, 18, 20, 22]
        L_HAND = [i for i in range(21)]
        R_HAND = [i for i in range(21)]
        
        # Extract selected columns
        FEATURE_COLUMNS = self.get_selected_features(l_hand=L_HAND, r_hand=R_HAND)

        # Extract indices of all features
        X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "x_" in col]
        Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "y_" in col]
        Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "z_" in col]

        # Extract indices of body part features
        # Hands
        RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "_right_hand_" in col]
        LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "_left_hand_" in col]
        # Pose
        RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in RPOSE]
        LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in LPOSE]

        if len(RHAND_IDX)//(len(RHAND_IDX)//3) % 3 != 0:
            raise ValueError("Error in RHAND_IDX dimension!")
        
        if len(LHAND_IDX)//(len(LHAND_IDX)//3) % 3 != 0:
            raise ValueError("Error in LHAND_IDX dimension!")

        if (len(RHAND_IDX) + len(LHAND_IDX) + len(RPOSE_IDX) + len(LPOSE_IDX)) != len(FEATURE_COLUMNS) and \
        (len(X_IDX) + len(Y_IDX) + len(Z_IDX)) != len(FEATURE_COLUMNS):
            raise ValueError('Error in extracting PROCESS_FINGER_SPELLING_TF indices!')
        
        return FEATURE_COLUMNS, RHAND_IDX, LHAND_IDX