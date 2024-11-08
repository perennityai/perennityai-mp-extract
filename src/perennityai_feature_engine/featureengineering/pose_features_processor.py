
import numpy as np
import tensorflow as tf

from perennityai_feature_engine.utils import IndiciesSpecification
from perennityai_feature_engine.utils import FeatureConfigLoader

import configparser
config = configparser.ConfigParser()
config.read('features_config.ini')


EUCLIDEAN_DISTANCE_LOW = config.getfloat('tf_constants', 'EUCLIDEAN_DISTANCE_LOW')
EUCLIDEAN_DISTANCE_HIGH = config.getfloat('tf_constants', 'EUCLIDEAN_DISTANCE_HIGH')

class PoseFeaturesProcessor:
    def __init__(self, logger=None):
        self.logger = logger
        # Expect (33, 3) body pose landmarks
        self.indices_map = {}

        self.body_pose_landmark_indices =  FeatureConfigLoader().load_body_pose_landmark_indices()
        self.pose_landmark_indices = self.get_indices_dict()
        


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
        all_indices = sorted({index for region in self.body_pose_landmark_indices.values() for part in region.values() for index in part})
        return all_indices
    
    def get_indices_dict(self):
        out = {}
        # Get master dict
        self.indices_map, POSE_IDX = self.get_indices_map()
        
        # self.indices_map contain x,y,z of each landmark points
        # indices_dist contain index of extracted data using POSE_IDX as key (map point to indices of landmarks in this class)
        # self.body_pose_landmark_indices contain original points in values()
        # We need to create pose_landmark_indices to have the x,y,z of indices_dist
        pose_landmark_indices = {}
        for idx in self.get_indices():
            if idx not in pose_landmark_indices.keys():
                pose_landmark_indices[idx] = self.indices_map.get(idx)

        # Create indices to find element in landmark
        indices_dist = dict(zip(POSE_IDX, list(range(0, len(POSE_IDX)))))

        # Iterate and assign indices for each region
        for key, value in pose_landmark_indices.items():
            # Flatten `value` if it contains lists within lists, then retrieve and flatten each `indices_dist.get(idx)` result
            out[key] = [val for idx in (item for item in value) for val in np.array(indices_dist.get(idx)).flatten()]
            #self.logger.debug(key, out[key])    
        return out
    
    def get_indices_map(self, hand='left'):
        hand_dict = {}
        # Get Extracted Columns
        FEATURE_COLUMNS, _, _, POSE_IDX, _ = IndiciesSpecification(logger=self.logger).process_dynamic_idx()

        # Iterate the extracted columns
        for i, col in enumerate(FEATURE_COLUMNS):
            if  f'_pose_' in col:
                idx = int(col[-2:].replace("e_", "").replace("_", ""))
                if idx in self.get_indices():
                    if not idx in hand_dict:
                        hand_dict[idx] = {}
                    # Store index of each coordinate
                    if "x_" in col and 'x' not in hand_dict[idx].keys():
                        name = 'x'
                        hand_dict[idx][name] = i
                    elif "y_" in col and 'y' not in hand_dict[idx].keys():
                        name = 'y'
                        hand_dict[idx][name] = i
                    elif "z_" in col and 'z' not in hand_dict[idx].keys():
                        name = 'z'
                        hand_dict[idx][name] = i
        
        # Convert coordinate dict to list
        hand_dict = {key:list(value.values()) for key, value in hand_dict.items()}

        return hand_dict, POSE_IDX
    
    def calculate_euclidean_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
            point1 (tf.Tensor): A tensor of shape (128, n, 3) representing the coordinates of the first set of points in the batch.
            point2 (tf.Tensor): A tensor of shape (128, n, 3) representing the coordinates of the second set of points in the batch.

        Returns:
            tf.Tensor: A tensor of shape (128, n, 3) containing the Euclidean distance between each corresponding pair of points in the batch.
        
        Description:
            This method calculates the Euclidean distance between each pair of points in `point1` and `point2` by first computing 
            the element-wise difference, then applying `tf.norm` with `axis=1` to obtain the L2 norm for each pair, and finally 
            retaining the original shape by setting `keepdims=True`.
        """
        # Calculate the Euclidean distance
        distance = tf.norm(point1 - point2, axis=1, keepdims=True)
        
        # # Use for debugging. It breaks the distance calculate steps down
        # # Subtract the tensors element-wise
        # diff = point1 - point2
        # print(diff.numpy().tolist())
        # print("-" * 30)
        # # Compute the squared differences
        # squared_diff = tf.square(diff)

        # # Sum the squared differences along the last axis (axis=1)
        # sum_squared_diff = tf.reduce_sum(squared_diff, axis=1)

        # # Take the square root to get the Euclidean distance
        # euclidean_distance = tf.sqrt(sum_squared_diff)

        return distance
    
    def extract_frame_features(self, landmarks,  normalize=True, is_left=False, average_out=False, enforce_indices=[]):
        """
        Extract features for a single frame by calculating distances
        between key body points dynamically based on body_pose_landmark_indices.
        
        Args:
            landmarks (tf.Tensor): Tensor of shape (33, 3) containing body pose landmarks.
        
        Returns:
            frame_features (dict): Dictionary with calculated distances between body parts.
        """
        frame_features = []

        #self.logger.debug("enforce_indices mp: ", enforce_indices)
        #self.logger.debug("landmarks shape : ", landmarks.shape)

        # Calculate a reference length for normalization (e.g., shoulder distance)
        left_shoulder_idx = self.body_pose_landmark_indices["torso"]["left_shoulder"][0]
        right_shoulder_idx = self.body_pose_landmark_indices["torso"]["right_shoulder"][0]

        # Get corresponding indices
        left_shoulder_idx = self.pose_landmark_indices[left_shoulder_idx] # [1,4,8]
        right_shoulder_idx = self.pose_landmark_indices[right_shoulder_idx] # [1,4,8]

        # Step 1: Gather the specified indices along axis 1
        left_shoulder_coords = tf.gather(landmarks, left_shoulder_idx, axis=1)  # Shape will be (119, 9)
        #self.logger.debug(left_shoulder_coords.shape, " left_shoulder_coords")

        right_shoulder_coords = tf.gather(landmarks, right_shoulder_idx, axis=1)  # Shape will be (119, 9)
        #self.logger.debug(right_shoulder_coords.shape, "right_shoulder_coords")
        
        # Step 2: Reshape to (119, 1, 3) to align with normalization_factor
        left_shoulder_coords = tf.reshape(left_shoulder_coords, (landmarks.shape[0], -1, 3))
        right_shoulder_coords = tf.reshape(right_shoulder_coords, (landmarks.shape[0], -1, 3))
        
        #self.logger.debug("left_shoulder_coords reshaped : ", left_shoulder_coords.shape)
        #self.logger.debug("right_shoulder_coords reshaped: ", right_shoulder_coords.shape)
        
        # Step 3: 
        reference_length = self.calculate_euclidean_distance(left_shoulder_coords, right_shoulder_coords)
        #self.logger.debug("reference_length ", reference_length.shape)

        # If normalization is enabled and reference length is valid, apply it
        if normalize and not tf.reduce_all(tf.equal(reference_length, 0)):
            normalization_factor = reference_length
        else:
            # No normalization if reference length is invalid
            # Create a tensor of ones with shape (128, 1) and dtype tf.float32
            normalization_factor = tf.ones((landmarks.shape[0], 1), dtype=tf.float32)

        # Step 5: Loop through body regions and calculate pairwise distances
        for region, parts in self.body_pose_landmark_indices.items():
            part_names = list(parts.keys())
            for i in range(len(part_names)):
                for j in range(i + 1, len(part_names)):
                    part1 = part_names[i]
                    part2 = part_names[j]
                    idx1 = parts[part1][0]
                    idx2 = parts[part2][0]

                    if not tf.equal(tf.size(enforce_indices), 0):
                        if not tf.reduce_any(tf.equal(enforce_indices, idx1)) and not tf.reduce_any(tf.equal(enforce_indices, idx2)):
                            #self.logger.debug("Skipping pose idx : ", idx1, " and ", idx2)
                            continue

                    # Step 6: Gather the specified indices along axis 1
                    point_coords_1 = tf.gather(landmarks, self.pose_landmark_indices[idx1], axis=1)  # Shape will be (119, 9)
                    point_coords_2 = tf.gather(landmarks, self.pose_landmark_indices[idx2], axis=1)  # Shape will be (119, 9)
                    
                    # Step 7: Reshape to (119, 6, 3) to align with normalization_factor
                    point_coords_1 = tf.reshape(point_coords_1, (landmarks.shape[0], -1, 3))
                    point_coords_2 = tf.reshape(point_coords_2, (landmarks.shape[0], -1, 3))

                    # Step 8: Extract x, y, z channels for mirroring if needed
                    x_coords_1 = point_coords_1[..., 0]  # Shape (119, 1)
                    y_coords_1 = point_coords_1[..., 1]  # Shape (119, 1)
                    z_coords_1 = point_coords_1[..., 2]  # Shape (119, 1)

                    # Step 9: Extract x, y, z channels for mirroring if needed
                    x_coords_2 = point_coords_2[..., 0]  # Shape (119, 1)
                    y_coords_2 = point_coords_2[..., 1]  # Shape (119, 1)
                    z_coords_2 = point_coords_2[..., 2]  # Shape (119, 1)

                    # Step 10: Apply mirroring on the x coordinates if is_left is True
                    if is_left:
                        x_coords_1 = 1 - x_coords_1
                        x_coords_2 = 1 - x_coords_2

                    # Step 11: Stack the coordinates back together and apply normalization
                    point_coords_1 = tf.stack([x_coords_1, y_coords_1, z_coords_1], axis=-1)
                    point_coords_2 = tf.stack([x_coords_2, y_coords_2, z_coords_2], axis=-1)

                    #self.logger.debug("reference_length : ", reference_length.shape)
                    #self.logger.debug("normalization_factor : ", normalization_factor.shape)
                    #self.logger.debug("point_coords_1 : ", point_coords_1.shape)
                    #self.logger.debug("point_coords_2 : ", point_coords_2.shape)

                    # Step 12: Normalize if specified
                    if normalize:
                        mean = tf.reduce_mean(point_coords_1, axis=0)
                        std = tf.math.reduce_std(point_coords_1, axis=0) + 1e-6  # To prevent division by zero
                        point_coords_1 = (point_coords_1 - mean) / std

                        mean = tf.reduce_mean(point_coords_2, axis=0)
                        std = tf.math.reduce_std(point_coords_2, axis=0) + 1e-6  # To prevent division by zero
                        point_coords_2 = (point_coords_2 - mean) / std
                    
                    #self.logger.debug("point_coords_1 norm : ", point_coords_1.shape)
                    #self.logger.debug("point_coords_2 norm : ", point_coords_2.shape)

                    # Step 13: Calculate euclidean distance
                    distance = self.calculate_euclidean_distance(point_coords_1, point_coords_2) 

                    # Step 14: Perform division
                    divided_features = distance / normalization_factor

                    # Step 15: Define the valid range
                    valid_range = (-tf.sqrt(EUCLIDEAN_DISTANCE_LOW), tf.sqrt(EUCLIDEAN_DISTANCE_HIGH))
                    # Check if any values are out of the valid range [-sqrt(2), sqrt(2)]
                    is_out_of_range = tf.reduce_any((divided_features < valid_range[0]) | (divided_features > valid_range[1]))

                    # Step 16: Apply conditional logic: if out of range, clip; if within range, use as is
                    # Ensure Proper Normalization Factor (e.g., 0 to 1 or standardized) using min-max scaling
                    # Set a minimum threshold for normalization_factor to avoid large divisions
                    region_features = tf.cond(
                        is_out_of_range,
                        lambda: tf.clip_by_value(distance / tf.where(normalization_factor < 1e-6, 1e-6, normalization_factor), valid_range[0], valid_range[1]),  # Alternative method with clamping
                        lambda: divided_features  # Use divided_features if all values are within range
                    )

                    #self.logger.debug("region_features : ", region_features.shape)
   
                    frame_features.append(region_features)

        pose =  tf.concat(frame_features, axis=1) 

        # Calculate the mean across the second dimension (axis=1)
        if average_out:
            pose = tf.reduce_mean(pose, axis=1, keepdims=True)

        #self.logger.debug("pose ", pose.numpy().tolist())
        #self.logger.debug("pose ", pose.shape)

        return pose
        