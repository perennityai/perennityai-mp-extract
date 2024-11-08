import sys
import os
import argparse


# Add the src directory to the module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from perennityai_feature_engine.preprocessing import Preprocessor

def main(args):
    print('args.input_dir : ', args.input_dir)
    print('args.output_dir : ', args.output_dir)
    print('args.processor : ', args.processor)
    print('args.extract : ', args.extract)
    print('args.extract_only : ', args.extract_only)
    print('args.padding : ', args.padding)
    print('args.average_out_face : ', args.average_out_face) 
    print('args.overwrite : ', args.overwrite) 
    print('args.pid : ', args.pid) 

    # Create a config dictionary from the arguments
    config = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "processor": args.processor,
        "extract": args.extract,
        "extract_only": args.extract_only,
        "padding": args.padding,
        "average_out_face": args.average_out_face,
        "overwrite": args.overwrite,
        "pid": args.pid
    }

    # Instantiate processor
    feature_processor = Preprocessor.from_pretrained(config=config)
    # Process Dataset
    feature_processor.process_dataset(padding=args.padding, extract_only=args.extract_only) # padding=True if training that data



if __name__ == "__main__":
    """
    Parses command-line arguments for specifying input and output directories.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing `input_dir` and `output_dir`.
    """
    parser = argparse.ArgumentParser(description="Feature Processor for gesture data.")

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing train_landmarks folder and train.csv file."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where processed features and logs will be saved."
    )

    parser.add_argument(
        "--processor",
        type=str,
        default='',
        required=False,
        help="The gesture processor to use for samples process."
    )

    parser.add_argument(
        "--extract",
        type=bool,
        default=False,
        required=False,
        help="The gesture processor to use for samples process."
    )

    parser.add_argument(
        "--extract_only",
        type=bool,
        default=False,
        required=False,
        help="Do not write selected columns, only write process features."
    )
    
    parser.add_argument(
        "--padding",
        type=bool,
        default=False,
        required=False,
        help="To pad and resize processed tfrecord samples to source_maxlen"
    )

    parser.add_argument(
        "--average_out_face",
        type=bool,
        default=False,
        required=False,
        help="To average all face landmarks to one. emg. (128,1,3) or (128,3)"
    )

    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        required=False,
        help="To overwrite generated tfrecord files"
    )

    parser.add_argument(
        "--pid",
        type=bool,
        default=0,
        required=False,
        help="Unique identification number of source of data"
    )

    # Process the data based on the specified processor type

    args = parser.parse_args()
    
    main(args)


"""
input_dir            : Directory path where input data (train_landmarks folder and train.csv) is located.
output_dir           : Directory path where processed features and logs will be saved.
extract              : It will extract data used for training, after there is no other processing needed
extract_only         : Do not write selected columns, only write process features in process_dataset() 
padding              : Pad extracted features in process_dataset()
average_out_face     : output avearage of all face points. e.g (128,1,3) or (128,3) 
overwrite            : Overwrite tfrecord files.
pid            : Unique identification number of source of data.
processor            : List of processors: (All are time series dataset). Leave empty if no selected columns proecessing
                        1. dynamic : dynamic gestures
                        2. fingerspelling: fingerspelling from internet data
                        3. kaggle_fingerspelling: fingerspelling from google kaggle data
                        4. static: only hands in a dynamic gesture
Example 1:
##########
processor : dynamic, static or fingerspelling

extract : ON
extract_only : ON
Result : will output one tfrecord (distance features)

Example 2:
##########
processor : dynamic, static or fingerspelling

extract : OFF
extract_only : OFF
Result:  one .tfrecords( selected features(raw))

Example 3:
##########
processor : dynamic, static or fingerspelling

extract : ON
extract_only : OFF
Result:  two files .tfrecords( selected features(raw) and distance features)
"""



if __name__ == "__main__":
    main()
