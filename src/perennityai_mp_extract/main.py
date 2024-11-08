import sys
import os
import argparse


# Add the src directory to the module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from perennityai_mp_extract.data_extraction import DataExtractor


def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Extractor Processor for creating MediaPipe landmarks dataset files.")
    
    # Input/Output arguments
    parser.add_argument(
        '--input_file', 
        type=str, 
        required=True, 
        default='', 
        help='Path to the input video file. Acceptable formats include mp4, avi, mov, ogg, etc.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help='Directory to save output files, such as extracted landmarks in the specified format.'
    )
    parser.add_argument(
        '--label', 
        type=str, 
        required=True, 
        help='Label or short target text associated with the landmarks in the video file.'
    )

    # Format and file handling
    parser.add_argument(
        '--output_format', 
        type=str, 
        choices=['csv', 'tfrecord', 'parquet'], 
        default='csv', 
        help='Output format for the landmarks file. Choices are "csv", "tfrecord", or "parquet".'
    )

    # Visualization options
    parser.add_argument(
        '--frame_rate', 
        type=int, 
        default=1, 
        help='Rate at which video frames are processed for landmark extraction. A lower frame rate can reduce processing time.'
    )

    # Logging and verbosity
    parser.add_argument(
        '--verbose', 
        type=str, 
        default='INFO', 
        choices=['DEBUG', 'ERROR', 'WARNING', 'INFO'], 
        help='Set logging level to control output verbosity for easier debugging.'
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    try:
        config = {
            'output_dir':args.output_dir,
            'output_format':args.output_format,
            'verbose':args.verbose
        }

        # Initialize the DataExtractor
        extractor = DataExtractor(config=config )

        extractor.extract(
            args.input_file,
            phrase=args.label
        )

          
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"Configuration loading failed: {e}")

if __name__ == "__main__":
    main()
