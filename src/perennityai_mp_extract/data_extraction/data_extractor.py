import os
import re
import cv2
import json
import pandas as pd
import numpy as np
import mediapipe
import yt_dlp
import mediapipe as mp
from urllib.parse import urlparse

from perennityai_mp_extract.utils import TFRecordProcessor
from perennityai_mp_extract.utils import CSVHandler
from perennityai_mp_extract.utils import Log

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Initialize MediaPipe models
mp_pose = mediapipe.solutions.pose
mp_hands = mediapipe.solutions.hands
mp_face_mesh = mediapipe.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation



class DataExtractor:
    """
    A class to extract data (e.g., landmarks, frames) from video files and save it in various formats.
    
    The class handles video extraction at a specified frame rate, saves data in the configured output format,
    and provides logging functionality for tracking the process.
    """

    def __init__(self, config=None):
        """
        Initializes the DataExtractor instance by setting configuration values and initializing necessary components.
        
        Args:
            config (dict, optional): A dictionary containing configuration options for extraction settings.
        """
        # Set the frame rate for data extraction, defaulting to 1 if not specified
        self.frame_rate = 1 if config.get("frame_rate") is None else config.get("frame_rate", 1)

        # Set the desired output format for saving extracted data (e.g., csv, json, tfrecord)
        self.output_format = config.get("output_format", '')

        # Set the directory where the output files will be saved; create it if it doesn't exist
        self.output_dir = config.get("output_dir", "")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize the logger to track the extraction process, storing logs in the output directory
        self.logger = Log(log_file=os.path.join(self.output_dir,  f"data-extractor.log"), verbose=config.get("verbose", "INFO"))

        # Initialize CSVHandler to handle CSV output (for saving data in CSV format)
        self.csv = CSVHandler(logger=self.logger)

        # Initialize TFRecordProcessor for handling TFRecord output (for saving data in TFRecord format)
        self.tf_processor = TFRecordProcessor(logger=self.logger)

        # Log the initialization of the DataExtractor instance
        self.logger.info("Initialized DataExtractor!")

    @classmethod
    def from_pretrained(cls, path: str):
        # Load configuration from a file (e.g., JSON format)
        config_path = os.path.join(path, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"No configuration file found at {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # Instantiate DataPreprocessor with the loaded configuration
        return cls(config=config)
    
    def get_video_id(self, youtube_url):
        """
        Extracts a sanitized video ID or filename from a given YouTube (or general video) URL.
        
        Args:
            youtube_url (str): The URL of the video from which the video ID or filename is to be extracted.

        Returns:
            str: The sanitized video ID or filename based on the URL.
        """
        
        def sanitize_filename(filename):
            """
            Sanitizes the filename by replacing invalid characters with underscores.
            
            Invalid characters include: '<', '>', ':', '"', '/', '\\', '|', '?', '*'.
            
            Args:
                filename (str): The original filename to sanitize.
                
            Returns:
                str: The sanitized filename with invalid characters replaced by underscores.
            """
            return re.sub(r'[<>:"/\\|?*]', '_', filename)

        def get_filename(link, index=0):
            """
            Generates a sanitized filename for a given video link by extracting its name and removing
            invalid characters.
            
            Args:
                link (str): The URL or path of the video.
                index (int, optional): The index to prepend to the filename. Defaults to 0.
                
            Returns:
                str: The sanitized filename with a numeric index and video title.
            """
            filename_str = f"{index}"  # Start with the index for the filename
            if link:
                if '.mp4' in link:
                    parsed_url = urlparse(link)  # Parse the URL to get the file name
                    file_name = os.path.basename(parsed_url.path)  # Extract the file name from the URL
                    video_file_word = sanitize_filename(file_name).replace('.mp4', '')  # Sanitize and remove the '.mp4' extension
                    sub_name = link.split('/')[-2]  # Extract a part of the URL to use as a sub-name
                    filename_str = '{}_{}'.format(sub_name, video_file_word)  # Combine the sub-name and sanitized file name
            return filename_str  # Return the generated filename
        
        # Check if the URL is not a YouTube URL
        if not "youtube" in youtube_url:
            # If it's not a YouTube URL, return the sanitized filename
            return get_filename(youtube_url)

     
    def fetch_video_metadata(self, video_url, format=None):
        """
        Fetches metadata for a given video URL using yt-dlp (a YouTube-dl fork).
        
        Args:
            video_url (str): The URL of the video to fetch metadata for.
            format (str, optional): The format specification for the video. 
                                    If None, it defaults to fetching the best quality video up to 1080p.

        Returns:
            dict: A dictionary containing the metadata of the video.
        """
        # Initialize an empty dictionary to store the video metadata
        info = {}

        # Default format option: 'best' video quality (up to 1080p) with best audio
        best = "bestvideo[height<=1080]+bestaudio/best"
        
        # If a custom format is provided, use it instead of the default
        if format is not None:
            best = format

        # yt-dlp options for extracting metadata without downloading the video
        ydl_opts = {
            'quiet': True,                  # Suppress output messages
            'no_warnings': True,            # Suppress warnings
            'skip_download': True,          # Skip actual download (only fetch metadata)
            'format': best,                 # Use the specified format for video selection
            'download_size': 360,           # Limit the download size (in MB)
            'download_audio_rate': 44100,   # Set the audio sample rate
            'yt_metadata_args': {
                'writeinfojson': True,      # Write metadata as a JSON file
                'force_generic_extractor': True,  # Force using a generic extractor
                'writesubtitles': 'all',    # Write all available subtitles
                'subtitlelangs': ['en'],    # Select English subtitles
                'get_info': True            # Retrieve detailed video information
            },
        }

        try:
            # Using yt-dlp to extract metadata without downloading the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
        except:
            # If an error occurs, log it and continue (do not crash the program)
            self.logger.debug("Error in extraction metadata")
            pass

        # Return the metadata dictionary
        return info

    
    def extract_landmarks(self, frame):
        """
        Extracts hand, face, and pose landmarks from a given frame using MediaPipe's hand, face, and pose models.

        Args:
            frame (numpy.ndarray): A single frame (image) from a video feed to process.

        Returns:
            tuple:
                - results (dict): A dictionary containing the landmarks for hands, face, and pose.
                - hand_results (object): Hand results from MediaPipe, including landmarks and handedness.
        """
        # Initialize MediaPipe hands, face mesh, and pose models for landmark detection
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands, \
            mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh, \
            mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:

            # Initialize a results dictionary to store landmarks for hands, face, and pose
            results = {
                'hands': [],  # List to store hand landmarks
                'face': [],   # List to store face landmarks
                'pose': []    # List to store pose landmarks
            }

            # Convert the frame to RGB, as MediaPipe models expect RGB images
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand landmarks using the MediaPipe hands model
            hand_results = hands.process(frame_rgb)
            if hand_results.multi_hand_landmarks:
                # If hands are detected, append the landmarks to the 'hands' list
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    results['hands'].append(hand_landmarks.landmark)

            # Process face landmarks using the MediaPipe face mesh model
            face_results = face_mesh.process(frame_rgb)
            if face_results.multi_face_landmarks:
                # If faces are detected, append the landmarks to the 'face' list
                for face_landmarks in face_results.multi_face_landmarks:
                    results['face'].append(face_landmarks.landmark)

            # Process pose landmarks using the MediaPipe pose model
            pose_results = pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                # If a pose is detected, append the landmarks to the 'pose' list
                results['pose'].append(pose_results.pose_landmarks.landmark)

            # Return the results dictionary containing the landmarks and the raw hand results
            return results, hand_results

    
    def landmarks_to_dict(self, landmarks, frame_number, hands, precision=5):
        """
        Converts the detected landmarks of a hand(s) into a dictionary format that is compatible with pandas DataFrame.
        
        Args:
            landmarks (list): A list of detected landmarks for the current frame.
            frame_number (int): The current frame number being processed.
            hands (list): A list of hands detected in the current frame.
            precision (int): The number of decimal places to round the landmarks coordinates to (default is 5).
        
        Returns:
            dict: A dictionary with the frame number and the landmarks of each hand, ready to be converted into a DataFrame.
        """
            
        data = {'frame': frame_number}

        # Hand landmarks
        for i, hand in enumerate(landmarks['hands']):
            hand_tag = hands[0].lower() +'_hand'
            if i == 1 and len(hands) == 2:
                hand_tag = hands[i].lower() +'_hand'

            for j, lm in enumerate(hand):
                data[f'x_{hand_tag}_{j}'] = round(lm.x, precision)
                data[f'y_{hand_tag}_{j}'] = round(lm.y, precision)
                data[f'z_{hand_tag}_{j}'] = round(lm.z, precision)
                
        #for compatibility purposes, inpu nan for missing hang
        if len(hands) == 1:
            if 'left' == hands[0].lower():
                 hand_tag = 'right_hand'
            elif 'right' == hands[0].lower():
                hand_tag = 'left_hand'
            for j, lm in enumerate(hand):
                data[f'x_{hand_tag}_{j}'] = np.nan
                data[f'y_{hand_tag}_{j}'] = np.nan
                data[f'z_{hand_tag}_{j}'] = np.nan
            

        # Face landmarks
        for i, face in enumerate(landmarks['face']):
            for j, lm in enumerate(face):
                data[f'x_face_{j}'] = round(lm.x, precision)
                data[f'y_face_{j}'] = round(lm.y, precision)
                data[f'z_face_{j}'] = round(lm.z, precision)

        # Pose landmarks
        for i, pose in enumerate(landmarks['pose']):
            for j, lm in enumerate(pose):
                data[f'x_pose_{j}'] = round(lm.x, precision)
                data[f'y_pose_{j}'] = round(lm.y, precision)
                data[f'z_pose_{j}'] = round(lm.z, precision)

        return data
    
    def get_filename(self, video_url):
            """
            Generates a filename for landmark data based on a given video file URL.
            
            This method checks the file extension of the provided video URL. If it matches a known video 
            extension (e.g., .mp4, .mov, .webm, .avi, .ogg), the extension is replaced with "._extension" 
            to create a unique filename for storing landmark data. If no matching extension is found, 
            "._extension" is appended to the original file name.
            
            Parameters:
            -----------
            video_url : str
                The path or URL of the video file for which the landmark filename will be generated.
                
            Returns:
            --------
            str
                The generated filename with the appropriate landmark file extension.
            
            """
            
            # Define the extensions to check
            extensions = [".mp4", ".mov", ".webm", ".avi", ".ogg"]
            
            # Replace extension if found
            for ext in extensions:
                if video_url.endswith(ext):
                    lm_file_name = video_url.replace(ext, "._extension")
                    return lm_file_name
            
            # Return default if no extension matched
            return video_url + "._extension"
    
    def save_metadata_to_json(self, 
                              video_metadata, 
                              video_url, 
                              start, 
                              end, 
                              output_file, 
                              clips,
                              phrase):
        """
            Saves metadata for a specific video segment to a JSON file.

            This function creates a structured JSON output containing the metadata of a video 
            segment, including details such as the video URL, start and end times, and any associated 
            phrases or clips.

            Args:
                video_metadata (dict): Metadata of the video such as resolution, duration, codec, etc.
                video_url (str): URL of the source video.
                start (float): Start time (in seconds) of the segment to be saved.
                end (float): End time (in seconds) of the segment to be saved.
                output_file (str): Path where the JSON file will be saved.
                clips (list): List of clips included in the video segment.
                phrase (str): A phrase or description associated with this video segment.

            Reference:
                WebVid Dataset: https://www.paperwithcode.com/dataset/webvid

            Returns:
                None: Writes the metadata to a JSON file specified by `output_file`.
            """

        metadata = {
            "description": "",
            "videoID": video_metadata.get("id", ""),
            "start": start,
            "end": end,
            "caption": phrase, # yt_meta_dict, info, title
            "url": video_url,
            "key": f"{phrase}_{start}_{end}",
            "status": "success",
            "error_message": None,
            "yt_meta_dict": {"info": video_metadata},
            "clips": clips
        }

        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        self.logger.info(f"Metadata saved to {output_file}")

    def extract(self, video_url, phrase=''):
        """
        Extracts relevant information from a video file based on the provided URL and phrase.

        This function processes the given video URL and validates the necessary inputs 
        (video URL and phrase). It is intended to be used for extracting video segments, 
        landmarks, or other related data.

        Args:
            video_url (str): URL or file path of the video to be processed.
            phrase (str, optional): A label or keyword associated with the video. Defaults to an empty string.

        Raises:
            ValueError: If the `video_url` is invalid or empty.
            ValueError: If the `phrase` is an empty string.

        Returns:
            dict: A dictionary containing extracted information, where additional details will be 
                populated depending on the specific use case of the `extract` function.

        Example:
            extract("http://example.com/video.mp4", "example phrase")
        """
        info = {}

        if not video_url:
            raise ValueError("Provide valid input video file!")
        
        if not phrase:
            raise ValueError("Empty string labels are not allowed!")

        # Open video file or url
        cap = cv2.VideoCapture(video_url)

        # Initialize video data variables
        height = 0  # The height of the current video frame
        width = 0   # The width of the current video frame
        frame_count = 0  # Total number of frames processed so far
        all_landmarks = []  # List to store the landmarks of all frames
        video_fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second (FPS) of the video using OpenCV

        clips = []  # List to store clips of the video if required
        current_frame_time = 0.0  # Variable to store the current timestamp (time) of the frame in seconds

        # Loop through each frame of the video
        while cap.isOpened():
            success, frame = cap.read()  # Read the next frame from the video

            if not success:  # If the frame was not read successfully, exit the loop
                break

            height, width = frame.shape[:2]  # Get the height and width of the current frame

            # Get the current timestamp in seconds (from the video capture's position in milliseconds)
            current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) // 1000.0

            # Process every nth frame based on the desired frame rate
            if frame_count % self.frame_rate == 0:
                hands = []  # List to store hand landmarks for the current frame
                landmarks, hand_results = self.extract_landmarks(frame)  # Extract landmarks and hand results from the current frame

            # Collect left and right hand indices
            if hand_results.multi_hand_landmarks:
                for idx, results in enumerate(hand_results.multi_hand_landmarks):
                    # Get the handedness label.
                    handedness = hand_results.multi_handedness[idx].classification[0].label
                    # confidence = hand_results.multi_handedness[idx].classification[0].score
                    if handedness not in hands:
                        hands.append(handedness)
                
                # We can crop image to start from where any of the hands is detected!
                landmarks_dict = self.landmarks_to_dict(landmarks, frame_count, hands)
                if landmarks_dict:
                    all_landmarks.append(landmarks_dict)

            # Increment frame count
            frame_count += 1

        file_name = self.get_filename(self.get_video_id(video_url))

        if all_landmarks:
            # Create a DataFrame from the all_landmarks list
            df = pd.DataFrame(all_landmarks)       

            # Check if the number of columns is 1630 (1629 landmarks + 1 frame number column)
            if len(df.columns) == 1630:  
                # Insert the 'phrase' as the first column in the DataFrame
                df.insert(0, 'phrase', phrase)
                # Initialize output file
                output_file = ''
                # Handle different output formats based on the user's specified output format
                if self.output_format == 'csv':
                    # Create a CSV file output path by replacing the extension
                    output_file = os.path.join(self.output_dir, file_name.replace("._extension", ".csv"))
                    # Write the DataFrame to a CSV file
                    self.csv.write_csv_file(df, output_file)
                
                elif self.output_format == 'parquet':
                    # Create a Parquet file output path by replacing the extension
                    output_file = os.path.join(self.output_dir, file_name.replace("._extension", ".parquet"))
                    # Write the DataFrame to a Parquet file
                    self.csv.write_parquet_file(df, output_file)
                
                elif self.output_format == 'tfrecord':
                    # Create a TFRecord file output path by replacing the extension
                    output_file = os.path.join(self.output_dir, file_name.replace("._extension", ".tfrecord"))
                    # Write the DataFrame to a TFRecord file
                    self.tf_processor.write_df_to_tfrecord(output_file, df)
                
                else:
                    # Raise an error if the output format is unsupported
                    raise ValueError(f"Output format not supported! {self.output_format}")

                # Log that the landmarks have been saved successfully
                self.logger.info(f"Landmarks saved to {output_file}")
            else:
                # Log a debug message if the number of columns is not as expected
                self.logger.debug(f"Invalid Landmarks of len:  {len(df.columns)}")
        else:
            # Log a debug message if all_landmarks is empty or None
            self.logger.debug('Landmark is empty : ', all_landmarks)

        # Process metadata
        info  = self.fetch_video_metadata(video_url) 

        if info:
            # Set the 'caption' field in the info dictionary to the value of 'phrase'
            info['caption'] = phrase
            
            # If the 'yt_meta_dict' key does not exist in the 'info' dictionary, initialize it
            if not 'yt_meta_dict' in info.keys():
                info['yt_meta_dict'] = {}  # Create the 'yt_meta_dict' key as an empty dictionary
                info['yt_meta_dict']['info'] = {}  # Create an 'info' key within 'yt_meta_dict'

            # Add or update the title within the 'yt_meta_dict' info section with the value of 'phrase'
            info['yt_meta_dict']['info']['title'] = phrase

            # Add the frame rate (fps) of the video to the metadata
            info['yt_meta_dict']['fps'] = video_fps

            # Add the width and height (resolution) of the video to the metadata
            info['yt_meta_dict']['width'] = width
            info['yt_meta_dict']['height'] = height

            # Format and store the resolution in the 'info' section as "width x height"
            info['yt_meta_dict']['info']["resolution"] = f"{width}x{height}"

            # Prepare the filename to save the metadata, replacing the extension with ".json"
            output_file = os.path.join(self.output_dir, file_name.replace("._extension", ".json"))

            # Save the metadata as a JSON file
            self.save_metadata_to_json(
                                info,  # The metadata to save
                                video_url,  # The video URL or path
                                0.0,  # Start time of the video (0.0 seconds)
                                current_frame_time,  # End time of the video (current frame time)
                                output_file,  # Path to the output file where metadata will be saved
                                clips,  # Additional clip information
                                phrase)  # The phrase used as the video caption or label

        cap.release()