import configparser

class FeatureConfigLoader:
    def __init__(self, config_file="features_config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)

    def get_section_data(self, section):
        """Parse a section into a dictionary of integer lists."""
        return {
            key: list(map(int, value.split(',')))
            for key, value in self.config[section].items()
        }

    def load_all_columns(self):
        """Load all feature columns from the 'all_features' section."""
        all_columns = self.config['all_features']['ALL_FEATURE_COLUMNS']
        all_columns = all_columns.split(" ") if not "\t" in all_columns else all_columns.split("\t")
        return all_columns

    def load_expression_landmark_indices(self):
        """Load expression landmark indices from the 'expression_landmark_indices' section."""
        return self.get_section_data("expression_landmark_indices")

    def load_body_pose_landmark_indices(self):
        """Load body pose landmark indices from multiple sections."""
        sections = ['head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
        body_pose_landmark_indices = {}
        
        for section in sections:
            body_pose_landmark_indices[section] = self.get_section_data(section)
        
        return body_pose_landmark_indices

    def load_hand_landmark_indices(self):
        """Load hand landmark indices from wrist and finger sections."""
        hand_landmark_indices = {'wrist': [int(self.config['wrist']['wrist'])]}
        sections = ['thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky_finger']
        
        for section in sections:
            hand_landmark_indices[section] = self.get_section_data(section)
        
        return hand_landmark_indices


# Example usage
if __name__ == "__main__":
    loader = FeatureConfigLoader()

    # Load and print the indices
    expression_landmark_indices = loader.load_expression_landmark_indices()
    print("Expression Landmark Indices:", expression_landmark_indices)

    body_pose_landmark_indices = loader.load_body_pose_landmark_indices()
    print("Body Pose Landmark Indices:", body_pose_landmark_indices)

    hand_landmark_indices = loader.load_hand_landmark_indices()
    print("Hand Landmark Indices:", hand_landmark_indices)
