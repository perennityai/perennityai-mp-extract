
import numpy as np
import tensorflow as tf

from perennityai_feature_engine.utils import FeatureConfigLoader
from perennityai_feature_engine.utils import IndiciesSpecification

import configparser
config = configparser.ConfigParser()
config.read('features_config.ini')


EUCLIDEAN_DISTANCE_LOW = config.getfloat('tf_constants', 'EUCLIDEAN_DISTANCE_LOW')
EUCLIDEAN_DISTANCE_HIGH = config.getfloat('tf_constants', 'EUCLIDEAN_DISTANCE_HIGH')

# Reference : https://github.com/k-m-irfan/simplified_mediapipe_face_landmarks
class FacialExpressionProcessor:
    def __init__(self, logger=None):
        """
            Expects (468, 3) facial landmarks
            Normalization: A common approach is to use a reference length within the face, such as:
                            The distance between the eyes (inter-eye distance).
                            The distance between the nose and mouth or left and right eye corners.
        """
        self.logger = logger
        
        # Load selected before calling get_indices_dict()
        self.selected_expression_landmarks_indices =  FeatureConfigLoader().load_expression_landmark_indices()
        self.expression_landmark_indices = self.get_indices_dict()
    
               
    def get_indices(self):
        """
        Extract all unique landmark indices from `body_pose_landmark_indices`.

        Description:
            This method iterates through each region in `body_pose_landmark_indices`, 
            which is assumed to be a nested dictionary structure. It collects all landmark indices from 
            each part in every region, ensuring uniqueness by using a set comprehension. 
            The resulting indices are then sorted in ascending order to provide a consistent output.

        Returns:
            list: A sorted list of unique indices across all regions and parts in `body_pose_landmark_indices`.
        """          
        all_indices = sorted([index for indices in self.selected_expression_landmarks_indices.values() for index in indices])
        return all_indices
    
    def get_indices_dict(self):
        """
        Generate a dictionary mapping selected landmark regions to their corresponding master indices.

        Description:
            This method creates a master dictionary `indices_dist` that maps each unique index from 
            `self.get_indices()` to a sequential range of integer values. Using this master index dictionary, 
            it then iterates over `self.selected_expression_landmarks_indices` to build a region-specific 
            dictionary (`out`) where each region key is mapped to a list of assigned indices. This facilitates 
            easy reference of indices for each region based on a consistent global ordering.

        Returns:
            dict: A dictionary where each key corresponds to a region name in `selected_expression_landmarks_indices`,
                and each value is a list of mapped indices for the respective region.
        """
        out={}
        # Get master dict
        indices_map, FACE_IDX = self.get_indices_map()
        
        # indices_map contain x,y,z of each landmark points
        # indices_dist contain index of extracted data using FACE_IDX as key (map point to indices of landmarks in this class)
        # self.selected_expression_landmarks_indices contain original points in values()
        # We need to create selected_expression_landmarks_indices to have the x,y,z of indices_dist
        selected_expression_landmarks_indices = {}
        for key, value in self.selected_expression_landmarks_indices.items():
            if key not in selected_expression_landmarks_indices.keys():
                selected_expression_landmarks_indices[key] = [indices_map.get(val) for val in value]

        # Create indices to find element in landmark
        indices_dist = dict(zip(FACE_IDX, list(range(0, len(FACE_IDX)))))

        # Iterate and assign indices for each region
        for key, value in selected_expression_landmarks_indices.items():
            # Flatten `value` if it contains lists within lists, then retrieve and flatten each `indices_dist.get(idx)` result
            out[key] = [val for idx in (item for sublist in value for item in sublist) for val in np.array(indices_dist.get(idx)).flatten()]
            #self.logger.debug(key, out[key])    
        return out
    
    def get_indices_map(self):
        face_dict = {}
        # Get Extracted Columns
        FEATURE_COLUMNS, _, _, _, FACE_IDX = IndiciesSpecification(logger=self.logger).process_dynamic_idx()
        # Iterate the extracted columns
        for i, col in enumerate(FEATURE_COLUMNS):
            if  "_face_" in col:
                idx = int(col[-3:].replace("e_", "").replace("_", ""))
                if idx in self.get_indices():
                    if not idx in face_dict:
                        face_dict[idx] = {}
                    # Store index of each coordinate
                    if "x_" in col and 'x' not in face_dict[idx].keys():
                        name = 'x'
                        face_dict[idx][name] = i
                    elif "y_" in col and 'y' not in face_dict[idx].keys():
                        name = 'y'
                        face_dict[idx][name] = i
                    elif "z_" in col and 'z' not in face_dict[idx].keys():
                        name = 'z'
                        face_dict[idx][name] = i
        # Convert coordinate dict to list
        face_dict = {key:list(value.values()) for key, value in face_dict.items()}

        return face_dict, FACE_IDX
    
    def get_enforced_indices(self, enforce_indices):
        enforce_set = set(enforce_indices)  # Convert to set for faster lookup
        out = [
            idx for value in self.selected_expression_landmarks_indices.values()
            for idx in value if idx in enforce_set
        ]
        return out
    
    def calculate_euclidean_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
            point1 (tf.Tensor): A tensor of shape (128, 3) representing the coordinates of the first set of points in the batch.
            point2 (tf.Tensor): A tensor of shape (128, 3) representing the coordinates of the second set of points in the batch.

        Returns:
            tf.Tensor: A tensor of shape (128, 1) containing the Euclidean distance between each corresponding pair of points in the batch.
        
        Description:
            This method calculates the Euclidean distance between each pair of points in `point1` and `point2` by first computing 
            the element-wise difference, then applying `tf.norm` with `axis=1` to obtain the L2 norm for each pair, and finally 
            retaining the original shape by setting `keepdims=True`.
        """
        distance = tf.norm(point1 - point2, axis=1, keepdims=True)
        return distance

    def calculate_reference_length(self, landmarks, normalize=False):
        """
        Calculate a reference length, such as the distance between the left and right eye centers.
        
        Args:
            landmarks (tf.Tensor): Tensor of shape (N, 3) containing all facial landmarks.
        
        Returns:
            float: Reference length for normalization.
        """
       # Extract points
        left_eye_landmarks = tf.gather(landmarks, self.expression_landmark_indices["left_eye"], axis=1)
        right_eye_landmarks = tf.gather(landmarks, self.expression_landmark_indices["right_eye"], axis=1)
        
        if left_eye_landmarks.shape[1] != len(self.expression_landmark_indices["left_eye"]):
            raise ValueError("left_eye its not equal!")
        
        if right_eye_landmarks.shape[1] != len(self.expression_landmark_indices["right_eye"]):
            raise ValueError("right_eye its not equal!")
        
        #self.logger.debug("left_eye_landmarks : ", left_eye_landmarks.shape)
        #self.logger.debug("right_eye_landmarks : ", right_eye_landmarks.shape)

        # Step 2: Reshape to (119, 1, 3) to align with region
        left_eye_landmarks = tf.reshape(left_eye_landmarks, (landmarks.shape[0], -1, 3))
        right_eye_landmarks = tf.reshape(right_eye_landmarks, (landmarks.shape[0], -1, 3))

        #self.logger.debug("left_eye_landmarks : ", left_eye_landmarks.shape)
        #self.logger.debug("right_eye_landmarks : ", right_eye_landmarks.shape)

        # Average them
        left_eye_center = tf.reduce_mean(left_eye_landmarks,  axis=1, keepdims=True)
        right_eye_center = tf.reduce_mean(right_eye_landmarks,  axis=1, keepdims=True)

        #self.logger.debug("left_eye_center : ", left_eye_center.shape)
        #self.logger.debug("right_eye_center : ", right_eye_center.shape)

        distance = self.calculate_euclidean_distance(left_eye_center, right_eye_center)

        #self.logger.debug("distance : ", distance.shape)

        return distance
    
    def process_landmarks_dist(self, landmarks, is_left=False, is_normalize=True):
        """
        Processes landmarks by splitting into x, y, z coordinates,
        optionally mirroring x-coordinates for left side, and normalizing.
        
        Args:
            landmarks (tf.Tensor): Tensor of shape (num_landmarks, 3) containing the coordinates.
            is_left (bool): Whether to mirror x-coordinates for left side.
            is_normalize (bool): Whether to normalize landmarks.
            
        Returns:
            np.ndarray: Processed landmarks with mirrored x-coordinates if is_left and normalized if is_normalize.
        """
        #  Splitting into x, y, z coordinates
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        z_coords = landmarks[:, 2]
        
        # Apply mirroring for the left side if required
        if is_left:
            x_coords = 1 - x_coords

        # Combine into a single tensor of shape (num_landmarks, 3)
        processed_landmarks = tf.stack([x_coords, y_coords, z_coords], axis=-1)

        # Normalize if specified
        if is_normalize:
            mean = tf.reduce_mean(processed_landmarks, axis=0)
            std = tf.math.reduce_std(processed_landmarks, axis=0) + 1e-6  # To prevent division by zero
            processed_landmarks = (processed_landmarks - mean) / std

        return processed_landmarks
    
    def extract_frame_features(self, landmarks, normalize=True, is_left=None, average_out=False, enforce_indices=[]):
        """
        Extracts all features from facial landmarks by processing each region.
        
        Args:
            landmarks (tf.Tensor): Tensor of shape (468, 3) containing all facial landmarks.
        
        Returns:
            dict: Dictionary with processed features for each region.
        """
        #self.logger.debug("face landmarks : ", landmarks.shape)
        #self.logger.debug("expression points : ", self.expression_landmark_indices)

         # Convert to class indices and tensor
        if enforce_indices:
            enforce_indices = self.get_enforced_indices(enforce_indices)
            #self.logger.debug("enforce_indices : ", enforce_indices) 

        if isinstance(enforce_indices, list):
            enforce_indices = tf.constant([enforce_indices])

        # Step 1: Gather the specified indices along axis 1
        nose_coords = tf.gather(landmarks, self.expression_landmark_indices["nose"], axis=1)  # Shape will be (119, 9)

        #self.logger.debug(nose_coords.shape, "nose_coords")

        # Step 2: Reshape to (119, 1, 3) to align with normalization_factor
        nose_coords = tf.reshape(nose_coords, (landmarks.shape[0], -1, 3))

        # Step 3: Calculate reference points using left and right eye points
        reference_length = self.calculate_reference_length(landmarks)

        # Step 4: If normalization is enabled and reference length is valid, apply it
        if normalize and not tf.reduce_all(tf.equal(reference_length, 0)):
            normalization_factor = reference_length
        else:
            # No normalization if reference length is invalid
            # Create a tensor of ones with shape (128, 1) and dtype tf.float32
            normalization_factor = tf.ones((landmarks.shape[0], 1, 3), dtype=tf.float32)

        all_features = []
        for region_name, indices in self.expression_landmark_indices.items():
            if not tf.equal(tf.size(enforce_indices), 0):
                # Find the intersection of indices and enforce_indices
                # Convert to sets, intersect, and convert back to a sorted tensor of integers
                enforce_indices = tf.constant(enforce_indices)
                filtered_indices = tf.sets.intersection(tf.expand_dims(indices, 0), tf.expand_dims(enforce_indices, 0))
                indices = tf.sparse.to_dense(filtered_indices)[0]

            # Mirror for left regions
            if is_left is not None:
                is_left = "left" in region_name 

            # Step 5: Gather the specified indices along axis 1
            gathered_coordinates = tf.gather(landmarks, indices, axis=1)  # Shape will be (119, 18)

            # Step 6: Reshape to (119, 6, 3) to align with normalization_factor
            gathered_coordinates_reshaped = tf.reshape(gathered_coordinates, (landmarks.shape[0], -1, 3))

            # Step 7: Extract x, y, z channels for mirroring if needed
            x_coords = gathered_coordinates_reshaped[..., 0]  # Shape (119, 6)
            y_coords = gathered_coordinates_reshaped[..., 1]  # Shape (119, 6)
            z_coords = gathered_coordinates_reshaped[..., 2]  # Shape (119, 6)

            # Step 8: Apply mirroring on the x coordinates if is_left is True
            if is_left:
                x_coords = 1 - x_coords

            # Step 9: Stack the coordinates back together and apply normalization
            point_coords = tf.stack([x_coords, y_coords, z_coords], axis=-1)

            #self.logger.debug("reference_length : ", reference_length.shape)
            #self.logger.debug("normalization_factor : ", normalization_factor.shape)
            #self.logger.debug("point_coords : ", point_coords.shape)

            # Step 10: Normalize if specified
            if normalize:
                mean = tf.reduce_mean(point_coords, axis=0)
                std = tf.math.reduce_std(point_coords, axis=0) + 1e-6  # To prevent division by zero
                point_coords = (point_coords - mean) / std

            #self.logger.debug("point_coords norm : ", point_coords.shape)

            # Step 11: Calculate euclidean distance
            distance = self.calculate_euclidean_distance(nose_coords, point_coords) 

            # Step 12: Perform division
            divided_features = distance / normalization_factor

            # Define the valid range
            valid_range = (-tf.sqrt(EUCLIDEAN_DISTANCE_LOW), tf.sqrt(EUCLIDEAN_DISTANCE_HIGH))
            # Step 13: Check if any values are out of the valid range [-sqrt(2), sqrt(2)]
            is_out_of_range = tf.reduce_any((divided_features < valid_range[0]) | (divided_features > valid_range[1]))

            # Step 14: Apply conditional logic: if out of range, clip; if within range, use as is
            # Ensure Proper Normalization Factor (e.g., 0 to 1 or standardized) using min-max scaling
            # Set a minimum threshold for normalization_factor to avoid large divisions
            region_features = tf.cond(
                is_out_of_range,
                lambda: tf.clip_by_value(distance / tf.where(normalization_factor < 1e-6, 1e-6, normalization_factor), valid_range[0], valid_range[1]),  # Alternative method with clamping
                lambda: divided_features  # Use divided_features if all values are within range
            )
        
            # Step 15: Calculate distance, mirror and normalize using standard deviation and reference point
            all_features.append(region_features)

        # Step 16: Combine all feature
        face =  tf.concat(all_features, axis=1)
        #self.logger.debug("face after concat : ", face.shape)

        # Step 17: Optional. Calculate the mean across the second dimension (axis=1)
        if average_out:
            face = tf.reduce_mean(face, axis=1, keepdims=True)

        #self.logger.debug("Out Face : ", face.numpy().tolist())
        #self.logger.debug("Out Face : ", face.shape)

        return face

