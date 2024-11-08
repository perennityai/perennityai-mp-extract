
import numpy as np
import tensorflow as tf

from perennityai_feature_engine.utils import IndiciesSpecification
from perennityai_feature_engine.utils import FeatureConfigLoader

import configparser
config = configparser.ConfigParser()
config.read('features_config.ini')

EUCLIDEAN_DISTANCE_LOW = config.getfloat('tf_constants', 'EUCLIDEAN_DISTANCE_LOW')
EUCLIDEAN_DISTANCE_HIGH = config.getfloat('tf_constants', 'EUCLIDEAN_DISTANCE_HIGH')

class HandFeaturesProcessor:
    def __init__(self, logger=None):

        """
        Normalize in HandFeaturesProcessor:
            A common normalization approach for hands is to use a reference distance within the hand, such as:

            The distance from the wrist to the middle finger MCP (metacarpophalangeal joint) as it provides a good scale reference.
            Alternatively, the distance from the wrist to the index finger MCP could also be used.
            """
        self.logger=logger

        self.hand_landmark_indices = FeatureConfigLoader().load_hand_landmark_indices()
        self.left_hand_landmark_indices = self.get_indices_dict(hand='left')
        self.right_hand_landmark_indices = self.get_indices_dict(hand='right')
        

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

        all_indices = [indices[0] if isinstance(indices, list) else value[0] for key, indices in self.hand_landmark_indices.items() for value in (indices.values() if not isinstance(indices, list) else [indices])]
        
        return all_indices
    
    def get_indices_dict(self, hand='left'):
        out = {}
        # Get master dict
        indices_map, HAND_IDX = self.get_indices_map(hand=hand)
        
        # indices_map contain x,y,z of each landmark points
        # indices_dist contain index of extracted data using HAND_IDX as key (map point to indices of landmarks in this class)
        # self.hand_landmark_indices contain original points in values()
        # We need to create hand_landmark_indices to have the x,y,z of indices_dist
        hand_landmark_indices = {}
        for idx in self.get_indices():
            if idx not in hand_landmark_indices.keys():
                hand_landmark_indices[idx] = indices_map.get(idx)

        # Create indices to find element in landmark
        indices_dist = dict(zip(HAND_IDX, list(range(0, len(HAND_IDX)))))

        # Iterate and assign indices for each region
        for key, value in hand_landmark_indices.items():
            # Flatten `value` if it contains lists within lists, then retrieve and flatten each `indices_dist.get(idx)` result
            out[key] = [val for idx in (item for item in value) for val in np.array(indices_dist.get(idx)).flatten()]
            #self.logger.debug(key, out[key])    
        return out
    
    def get_indices_map(self, hand='left'):
        hand_dict = {}
        which_hand = ''
        indices = []
        # Get Extracted Columns
        FEATURE_COLUMNS, RHAND_IDX, LHAND_IDX, _, _ = IndiciesSpecification(logger=self.logger).process_dynamic_idx()

        # Assign values for left or right hand
        if hand == 'right':
            which_hand = 'right'
            indices = RHAND_IDX
        elif hand == 'left':
            which_hand = 'left'
            indices = LHAND_IDX

        # Iterate the extracted columns
        for i, col in enumerate(FEATURE_COLUMNS):
            if  f'_{which_hand}_hand_' in col:
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

        return hand_dict, indices

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
        
        # Use for debugging. It breaks the steps
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
    
    def extract_frame_features(self, landmarks, normalize=True, is_left=False, average_out=False, enforce_indices=[]):
        """
        Extract features for a single frame by calculating distances
        from the wrist to other key hand points.
        
        Args:
            landmarks (np.ndarray): Array of shape (21, 3) containing hand landmarks.
            normalize (bool): Whether to normalize distances based on a reference length.
        
        Returns:
            frame_features (dict): Dictionary with distances from the wrist to other key points,
                                   normalized if specified.
        """
        frame_features = []

        #self.logger.debug("hand landmarks : ", landmarks.shape)

        hand_landmark_indices = {}
        if not is_left:
            hand_landmark_indices = self.left_hand_landmark_indices
        else:
            hand_landmark_indices = self.right_hand_landmark_indices

        #self.logger.debug("hand_landmark_indices : ", hand_landmark_indices)

        # Reference length for normalization: Distance from wrist to middle finger MCP
        wrist_idx = self.hand_landmark_indices["wrist"][0]
        middle_finger_mcp_idx = self.hand_landmark_indices["middle_finger"]['mcp'][0]

        # Get corresponding indices
        wrist_idx = hand_landmark_indices[wrist_idx] # [1,4,8]
        middle_finger_mcp_idx = hand_landmark_indices[middle_finger_mcp_idx] # [1,4,8]

        # Step 1: Gather the specified indices along axis 1
        wrist_coords = tf.gather(landmarks, wrist_idx, axis=1)  # Shape will be (119, 9)
        #self.logger.debug(wrist_coords.shape, " wrist_coords")

        middle_finger_mcp_coords = tf.gather(landmarks, middle_finger_mcp_idx, axis=1)  # Shape will be (119, 9)
        #self.logger.debug(middle_finger_mcp_coords.shape, "middle_finger_mcp_coords")
        
        # Step 2: Reshape to (119, 1, 3) to align with normalization_factor
        wrist_coords = tf.reshape(wrist_coords, (landmarks.shape[0], -1, 3))
        middle_finger_mcp_coords = tf.reshape(middle_finger_mcp_coords, (landmarks.shape[0], -1, 3))
        
        #self.logger.debug("wrist_coords reshaped : ", wrist_coords.shape)
        #self.logger.debug("middle_finger_mcp_coords reshaped: ", middle_finger_mcp_coords.shape)
        
        # Step 3: 
        reference_length = self.calculate_euclidean_distance(wrist_coords, middle_finger_mcp_coords)

        #self.logger.debug("reference_length ", reference_length.shape)

        # If normalization is enabled and reference length is valid, apply it
        if normalize and not tf.reduce_all(tf.equal(reference_length, 0)):
            normalization_factor = reference_length
        else:
            # No normalization if reference length is invalid
            # Create a tensor of ones with shape (128, 1) and dtype tf.float32
            normalization_factor = tf.ones((landmarks.shape[0], 1, 3), dtype=tf.float32)
            
        # Step 4: Calculate distances from the wrist to each other landmark, and normalize if required
        for parent_key, indices in self.hand_landmark_indices.items():
            if parent_key == "wrist":
                wrist_init = tf.zeros((landmarks.shape[0], 1, 3), dtype=tf.float32)
                frame_features.append(wrist_init) # Otherwise, sample will not pass num_features * channels % 3 == 0 validation
                continue  # Skip the wrist itself

            # Use tf.sets.intersection to get only those indices in enforce_indices
            if not tf.equal(tf.size(tf.constant(enforce_indices)), 0):
                #self.logger.debug("enforce_indices mp: ", enforce_indices)
                # Extract corresponding indices
                indices = {key: [idx[0]] for key, idx in indices.items() if idx[0] in enforce_indices}
                #self.logger.debug("enforce_indices lm: ", indices)

            # Step 5: Loop through and process points 
            for key, idx in indices.items():
                # Extract points
                # Step 6: Gather the specified indices along axis 1
                point_coords = tf.gather(landmarks, hand_landmark_indices[idx[0]], axis=1)  # Shape will be (119, 9)

                # Step 7: Reshape to (119, 6, 3) to align with normalization_factor
                point_coords = tf.reshape(point_coords, (landmarks.shape[0], -1, 3))

                # Step 8: Extract x, y, z channels for mirroring if needed
                x_coords = point_coords[..., 0]  # Shape (119, 1)
                y_coords = point_coords[..., 1]  # Shape (119, 1)
                z_coords = point_coords[..., 2]  # Shape (119, 1)

                # Step 9: Apply mirroring on the x coordinates if is_left is True
                if is_left:
                    x_coords = 1 - x_coords
                    
                # Step 10: Stack the coordinates back together and apply normalization
                region_features = tf.stack([x_coords, y_coords, z_coords], axis=-1)

                #self.logger.debug("reference_length : ", reference_length.shape)
                #self.logger.debug("normalization_factor : ", normalization_factor.shape)
                #self.logger.debug("region_features : ", region_features.shape)

                # Step 11: Normalize if specified
                if normalize:
                    mean = tf.reduce_mean(region_features, axis=0)
                    std = tf.math.reduce_std(region_features, axis=0) + 1e-6  # To prevent division by zero
                    region_features = (region_features - mean) / std
                    
                #self.logger.debug("region_features norm : ", region_features.shape)

                # Step 12: Calculate euclidean distance
                distance = self.calculate_euclidean_distance(wrist_coords, point_coords) 

                # Step 13: Perform division
                divided_features = distance / normalization_factor

                # Step 14: Check if any values are out of the valid rangeDefine the valid range
                valid_range = (-tf.sqrt(EUCLIDEAN_DISTANCE_LOW), tf.sqrt(EUCLIDEAN_DISTANCE_HIGH))
                # Check if any values are out of the valid range [-sqrt(2), sqrt(2)]
                is_out_of_range = tf.reduce_any((divided_features < valid_range[0]) | (divided_features > valid_range[1]))

                # Step 15: Apply conditional logic: if out of range, clip; if within range, use as is
                # Ensure Proper Normalization Factor (e.g., 0 to 1 or standardized) using min-max scaling
                # Set a minimum threshold for normalization_factor to avoid large divisions
                region_features = tf.cond(
                    is_out_of_range,
                    lambda: tf.clip_by_value(distance / tf.where(normalization_factor < 1e-6, 1e-6, normalization_factor), valid_range[0], valid_range[1]),  # Alternative method with clamping
                    lambda: divided_features  # Use divided_features if all values are within range
                )
   
                #self.logger.debug(f"wrist_to_{key}_{idx}")

                # Step 16: Append to feature list
                frame_features.append(region_features)

        hand =  tf.concat(frame_features, axis=1) # Total of 63
        #self.logger.debug("Concat hand : ", hand.shape)

        # Calculate the mean across the second dimension (axis=1)
        if average_out:
            hand = tf.reduce_mean(hand, axis=1, keepdims=True)
        
        #self.logger.debug("Out Hand : ", hand.numpy().tolist())
        #self.logger.debug("Out Hand : ", hand.shape)

        return hand
        
