import os
import glob
import pandas as pd
import numpy as np
from typing import Optional
import tensorflow as tf

from .facial_features_processor import FacialExpressionProcessor
from .hand_features_processor import HandFeaturesProcessor
from .pose_features_processor import  PoseFeaturesProcessor

import configparser
config = configparser.ConfigParser()
config.read('features_config.ini')

SOURCE_MAX_LEN = config.getint('tf_constants', 'SOURCE_MAX_LEN')
AVERAGE_FACE_FEATURES = config.getint('tf_constants', 'AVERAGE_FACE_FEATURES')
AVERAGE_POSE_FEATURES = config.getint('tf_constants','AVERAGE_POSE_FEATURES')
AVERAGE_HAND_FEATURES = config.getint('tf_constants', 'AVERAGE_HAND_FEATURES')
VERBOSE = config['tf_constants']['VERBOSE']


class LandmarkFeatureExtractor:
    """
    total_num_features = 256 when AVERAGE_FACE_FEATURES = 0
    total_num_features = 154 when AVERAGE_FACE_FEATURES = 1
    """
    def __init__(self, indicies_spec: object, logger: object, average_out_face: Optional[bool] = False):
        """
        Initializes the class with specified indices, a logger, and an optional face averaging setting.

        Args:
            indicies_spec (dict): A dictionary specifying indices for feature extraction or processing.
            logger (object): Logger object for logging messages. This parameter is required.
            average_out_face (Optional[bool]): Determines whether to average out face landmarks. Default is False.
        """
        self.logger = logger
        self.average_out_face = average_out_face if average_out_face is not None else bool(AVERAGE_FACE_FEATURES)

        self.face_pro = FacialExpressionProcessor(logger = self.logger)
        self.hand_pro = HandFeaturesProcessor(logger = self.logger)
        self.pose_pro = PoseFeaturesProcessor(logger = self.logger)
        
        self.indicies_spec = indicies_spec
        self.source_maxlen = SOURCE_MAX_LEN
        
    # Function to resize and add padding.
    def resize_pad(self, x):
        # Determine the rank of x
        pad_idx = 1
        rank = tf.rank(x)
        shape = x.shape.as_list() 

        if tf.equal(rank, 2) or tf.equal(rank, 0) and len(shape) == 3: # Aslfingerspelling dataset
            if shape[0] < self.source_maxlen:
                x = tf.pad(x, ([[0, self.source_maxlen - shape[0]], [0, 0], [0, 0]]))
            else:
                # Truncate to the first `self.source_maxlen` rows
                x = x[:, :self.source_maxlen, :]
            #self.logger.debug("Resize pad is rank 2")
        elif tf.equal(rank, 3): # ggmt dataset
            if shape[pad_idx] < self.source_maxlen:
                # Pad the first dimension for 3D tensor
                x = tf.pad(x, [[0, 0], [0, self.source_maxlen - shape[pad_idx]], [0, 0]])
            elif shape[pad_idx] == self.source_maxlen:
                #self.logger.debug("resized is equal")
                pass
            else:
                # Truncate to the first `self.source_maxlen` rows
                x = x[:, :self.source_maxlen, :]
                #self.logger.debug("Truncated resize pad", x.shape)
                
        return x

    def process_dynamic_gesture(self, x: tf.Tensor, pad_and_resize: bool = False, average_out_face: Optional[bool] = None) -> tf.Tensor:
        """
        Processes dynamic gesture data, with options for padding, resizing, and averaging out face landmarks.

        Args:
            x (tf.Tensor): The input tensor representing gesture data.
            pad_and_resize (bool): If True, applies padding and resizing to the input tensor. Default is True.
            average_out_face (Optional[bool]): If True, averages out face landmarks; if None, uses the default 
                                            setting from self.average_out_face.

        Returns:
            tf.Tensor: Processed tensor after applying padding, resizing, and optional face averaging.
        """
        # Override average_out_face if specified, otherwise use class default
        this_average_out_face = average_out_face if average_out_face is not None else self.average_out_face

            
        if tf.size(x) < 1:
                raise ValueError("x cannot be empty!")
        
        # Replace all NaNs with 0 before splitting into parts
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        
        # Processing _face 423, _hand_ 126, _pose 99 = 648 columns
        # Get indicies
        _, RHAND_IDX, LHAND_IDX, POSE_IDX, FACE_IDX = self.indicies_spec.process_dynamic_idx()
        
        #self.logger.debug("FACE_IDX : ", len(FACE_IDX))
        #self.logger.debug("FACE_IDX : ", FACE_IDX)
        
        # Extract hand landmarks using predefined indices for right and left hands, poses and face
        #self.logger.debug("x : ", x.shape)
        rhand_landmarks = tf.gather(x, RHAND_IDX, axis=1)
        #self.logger.debug("rhand_features : ", rhand_landmarks.shape)
        lhand_landmarks = tf.gather(x, LHAND_IDX, axis=1)
        #self.logger.debug("lhand_landmarks : ", lhand_landmarks.shape)
        pose_landmarks = tf.gather(x, POSE_IDX, axis=1)
        #self.logger.debug("pose_landmarks : ", pose_landmarks.shape)
        face_landmarks = tf.gather(x, FACE_IDX, axis=1)
        #self.logger.debug("face_landmarks : ", face_landmarks.shape)

        """
        Mirror if you want to treat left and right sides similarly to enhance symmetry in training data.
        Do not mirror if you need to maintain distinctions between left and right-side poses or gestures.
        For normalization or alignment, it depends on whether other steps (like relative alignment) have 
        already standardized the orientation.
        """
        is_left = False # Left and right side gestures are different for GGMT intelligence purposes
        
        # Extract features
        lhand_features = self.hand_pro.extract_frame_features(lhand_landmarks, is_left=is_left, average_out=bool(AVERAGE_HAND_FEATURES))
        rhand_features = self.hand_pro.extract_frame_features(rhand_landmarks,  is_left=is_left, average_out=bool(AVERAGE_HAND_FEATURES))
        

        #self.logger.debug("rhand_features : ", rhand_features.shape)
        #self.logger.debug("lhand_features : ", lhand_features.shape)
        
        # Concatenate hands
        hand_features = tf.concat([rhand_features, lhand_features], axis=1)
        #self.logger.debug("hand_features : ", hand_features.shape)

        # Extract pose features
        pose_features = self.pose_pro.extract_frame_features(pose_landmarks, is_left=is_left, average_out=bool(AVERAGE_POSE_FEATURES)) # 
        #self.logger.debug("pose_features reshaped: ", pose_features.shape)

        # Extract face features
        face_features = self.face_pro.extract_frame_features(face_landmarks, is_left=is_left, average_out=this_average_out_face)
        #self.logger.debug("face_features : ", face_features.shape)

        # Reshape face features
        #self.logger.debug("face_features reshaped: ", face_features.shape)

        features_num_features = hand_features.shape[1] + pose_features.shape[1] + face_features.shape[1] 
        #self.logger.debug("features_num_features : ", features_num_features)

        x = tf.concat([hand_features, pose_features, face_features], axis=1)
        #self.logger.debug("features : ", x.shape)

        # Reshape to 2D
        x = tf.reshape(x, (-1, x.shape[1] * x.shape[2])) # (e.g (119, 462))
        
        if pad_and_resize:
            # Add batch dimension to match with existing processing
            x = tf.expand_dims(x, axis=0) # or x[tf.newaxis, ...] (1, 119, 462)
            #self.logger.debug("features expand_dim : ", x.shape)

            # Resize or pad to the target length
            x = self.resize_pad(x)
            #self.logger.debug("features resized : ", x.shape)

            x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
            #self.logger.debug("features removed NaNs: ", x.shape)

            x = tf.reshape(x, (self.source_maxlen, x.shape[2])) # (128, 154)
            #self.logger.debug("features reshaped: ", x.shape)

        return x

    def process_fingerspelling_gesture(self, x, pad_and_resize=False):
        if tf.size(x) < 1:
                raise ValueError("x cannot be empty!")
        
        # Processing _face 423, _hand_ 126, _pose 99 = 648 columns
        # Get indicies
        RHAND, LHAND, RPOSE, LPOSE = self.indicies_spec.get_finger_spelling_indices()
        _, RHAND_IDX, LHAND_IDX, _, _ = self.indicies_spec.process_fingerspelling_idx()
       
        
        # Extract hand landmarks using predefined indices for right and left hands, poses and face
        #self.logger.debug("x : ", x.shape)
        rhand_landmarks = tf.gather(x, RHAND_IDX, axis=1)
        #self.logger.debug("rhand_features : ", rhand_landmarks.shape)
        lhand_landmarks = tf.gather(x, LHAND_IDX, axis=1)
        #self.logger.debug("lhand_landmarks : ", lhand_landmarks.shape)

        # Get NaNs index
        rnan_idx = tf.reduce_any(tf.math.is_nan(rhand_landmarks), axis=1)
        lnan_idx = tf.reduce_any(tf.math.is_nan(lhand_landmarks), axis=1)

        # Count NaNs
        rnans = tf.math.count_nonzero(rnan_idx)
        lnans = tf.math.count_nonzero(lnan_idx)

        POSE = []
        HAND = []
        is_left = False  # Left and right side gestures are different for GGMT intelligence purposes
        if rnans > lnans:
            is_left = True 
            POSE = LPOSE
            HAND = RHAND
        else:
            POSE = RPOSE
            HAND = LHAND

        # Replace all NaNs with 0 before splitting into parts
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        
        # Extract features
        hand_features = self.hand_pro.extract_frame_features(x, is_left=is_left, enforce_indices=HAND, average_out=bool(AVERAGE_HAND_FEATURES))
        #self.logger.debug("hand_features : ", hand_features.shape)
        
        # Extract pose features using enforced position index assigned
        pose_features = self.pose_pro.extract_frame_features(x, is_left=is_left, enforce_indices=POSE, average_out=bool(AVERAGE_POSE_FEATURES)) #
        #self.logger.debug("pose_features: ", pose_features.shape)

        # Combine landmarks
        x = tf.concat([hand_features, pose_features], axis=1)
        #self.logger.debug("features : ", x.shape)

        # Reshape to 2D
        x = tf.reshape(x, (-1, x.shape[1] * x.shape[2])) # (e.g (119, 462))
        #self.logger.debug("features reshaped: ", x.shape)
        
        if pad_and_resize:
            # Add batch dimension to match with existing processing
            x = tf.expand_dims(x, axis=0) # or x[tf.newaxis, ...] (1, 119, 462)
            #self.logger.debug("features expand_dim : ", x.shape)

            # Resize or pad to the target length
            x = self.resize_pad(x)
            #self.logger.debug("features resized : ", x.shape)

            x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
            #self.logger.debug("features removed NaNs: ", x.shape)

            x = tf.reshape(x, (self.source_maxlen, x.shape[2])) # (128, 154)
            #self.logger.debug("features reshaped: ", x.shape)

        return x
    
    def process_static_gesture(self, x, pad_and_resize=False):
        if tf.size(x) < 1:
            raise ValueError("x cannot be empty!")
        # Get indicies
        _, RHAND_IDX, LHAND_IDX = self.indicies_spec.process_static_idx()

        # Replace all NaNs with 0 before splitting into parts
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

        # Extract hand landmarks using predefined indices for right and left hands and poses
        rhand_landmarks = tf.gather(x, RHAND_IDX, axis=1)  # Right hand landmarks
        #self.logger.debug("rhand_features : ", rhand_landmarks.shape)
        lhand_landmarks = tf.gather(x, LHAND_IDX, axis=1)  # Left hand landmarks
        #self.logger.debug("lhand_landmarks : ", lhand_landmarks.shape)

        # Extract features
        rhand_features = self.hand_pro.extract_frame_features(rhand_landmarks) # normalized
        #self.logger.debug("rhand_features : ", rhand_features.shape)
        lhand_features = self.hand_pro.extract_frame_features(lhand_landmarks) # normalized and don't invert
        #self.logger.debug("lhand_features : ", lhand_features.shape)

        # Concatenate all features into one tensor
        x = tf.concat([rhand_features, lhand_features], axis=1)
        #self.logger.debug("x : ", x.shape)

        # Reshape to 2D
        x = tf.reshape(x, (-1, x.shape[1] * x.shape[2])) # (e.g (119, 462))
        
        if pad_and_resize:
            # Add batch dimension to match with existing processing
            x = tf.expand_dims(x, axis=0) # or x[tf.newaxis, ...] (1, 119, 462)
            #self.logger.debug("features expand_dim : ", x.shape)

            # Resize or pad to the target length
            x = self.resize_pad(x)
            #self.logger.debug("features resized : ", x.shape)

            x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
            #self.logger.debug("features removed NaNs: ", x.shape)

            x = tf.reshape(x, (self.source_maxlen, x.shape[2])) # (128, 154)
            #self.logger.debug("features reshaped: ", x.shape)

        return x

    def process_kaggle_fingerspelling_gesture(self, x, pad_and_resize=False):
        #self.logger.debug("-> x : ", x.shape)

        def resize_pad(x):
            # Check if the sequence length is less than the maximum length, and pad if needed
            if tf.shape(x)[0] < self.source_maxlen:
                x = tf.pad(x, ([[0, self.source_maxlen - tf.shape(x)[0]], [0, 0], [0, 0]]))
            else:
                # Resize the tensor if it exceeds the maximum length
                x = tf.image.resize(x, (self.source_maxlen, tf.shape(x)[1]))
            return x
        
        def process_landmarks(landmarks, is_left=False, is_normalize=True):
            # Split into x, y, z coordinates
            x_coords = landmarks[:, 0 * (len(landmarks[0]) // 3): 1 * (len(landmarks[0]) // 3)]
            y_coords = landmarks[:, 1 * (len(landmarks[0]) // 3): 2 * (len(landmarks[0]) // 3)]
            z_coords = landmarks[:, 2 * (len(landmarks[0]) // 3): 3 * (len(landmarks[0]) // 3)]
            
            # Apply mirroring for the left hand only
            if is_left:
                x_coords = 1 - x_coords

            # Combine x, y, z into a tensor and normalize
            part = tf.concat([x_coords[..., tf.newaxis], y_coords[..., tf.newaxis], z_coords[..., tf.newaxis]], axis=-1)

            # Apply normalization. Normalize the coordinates by subtracting mean and dividing by standard deviation
            if is_normalize:
                mean = tf.math.reduce_mean(part, axis=1, keepdims=True)
                std = tf.math.reduce_std(part, axis=1, keepdims=True)
                part = (part - mean) / std

            return part   
        
        # Extract landmark indices
        _, RHAND_IDX, LHAND_IDX, RPOSE_IDX, LPOSE_IDX = self.indicies_spec.process_fingerspelling_idx(kaggle=True)

        # Detect the dominant hand from the number of NaN values.
        # Dominant hand will have less NaN values since it is in frame moving.
        rhand = tf.gather(x, RHAND_IDX, axis=1)
        lhand = tf.gather(x, LHAND_IDX, axis=1)
        rpose = tf.gather(x, RPOSE_IDX, axis=1)
        lpose = tf.gather(x, LPOSE_IDX, axis=1)

        #self.logger.debug("-> rhand : ", rhand.shape)
        #self.logger.debug("-> lhand : ", lhand.shape)
        #self.logger.debug("-> rpose : ", rpose.shape)
        #self.logger.debug("-> lpose : ", lpose.shape)

        # Identify NaN values across right and left hands to detect the dominant hand
        rnan_idx = tf.reduce_any(tf.math.is_nan(rhand), axis=1)  # NaN presence for right hand
        lnan_idx = tf.reduce_any(tf.math.is_nan(lhand), axis=1)  # NaN presence for left hand

        # Count the number of NaN values to determine the dominant hand
        rnans = tf.math.count_nonzero(rnan_idx)
        lnans = tf.math.count_nonzero(lnan_idx)

        # Determine the dominant hand based on fewer NaNs
        if rnans > lnans:
            hand = lhand  # Set left hand as dominant
            pose = lpose  # Set left pose as dominant

            # Split landmarks for x, y, z coordinates and adjust x-axis mirroring for left hand
            hand = process_landmarks(hand, is_left=True)
        else:
            # Use right hand and right pose as dominant
            hand = rhand
            pose = rpose

        # Reshape hand coordinates into (num_points, 3) format without further modification
        hand = process_landmarks(hand)
        #self.logger.debug("hand : ", hand.shape)

        # Reshape pose coordinates to (num_points, 3) format
        pose = process_landmarks(pose)
        #self.logger.debug("pose : ", pose.shape)

        # Concatenate hand and pose features along the spatial dimension
        x = tf.concat([hand, pose], axis=1)
        #self.logger.debug("x concat : ", x.shape)

        # Reshape to 2D
        x = tf.reshape(x, (-1, x.shape[1] * x.shape[2])) # (e.g (119, 462))

        if pad_and_resize:
            # Add batch dimension to match with existing processing
            x = tf.expand_dims(x, axis=0) # or x[tf.newaxis, ...] (1, 119, 462)
            #self.logger.debug("features expand_dim : ", x.shape)

            # Resize or pad to the target length
            x = resize_pad(x)
            #self.logger.debug("features resized : ", x.shape)

            x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
            #self.logger.debug("features removed NaNs: ", x.shape)

            x = tf.reshape(x, (self.source_maxlen, x.shape[2])) # (128, 154)
            #self.logger.debug("features reshaped: ", x.shape)

        return x  # Return the preprocessed tensor


