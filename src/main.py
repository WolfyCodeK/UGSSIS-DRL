import argparse
import sys
from pathlib import Path
import src.config as config

def main():
    parser = argparse.ArgumentParser(description="UGS Segmentation Pipeline")
    
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Run data preprocessing'
    )
    
    parser.add_argument(
        '--train', 
        action='store_true', 
        help='Run the active learning training loop'
    )
    
    parser.add_argument(
        '--evaluate', 
        type=str,
        default=None,
        metavar='path\\to\\model',
        help='Evaluate a trained model checkpoint'
    )

    parser.add_argument(
        "--test",
        action='store_true',
        help="Run the test loop"
    )

    # Check for --evaluate without argument before argparse processes it
    if '--evaluate' in sys.argv and len(sys.argv) > 1:
        eval_index = sys.argv.index('--evaluate')
        if eval_index == len(sys.argv) - 1 or (eval_index + 1 < len(sys.argv) and sys.argv[eval_index + 1].startswith('-')):
            parser.print_usage()
            print(f"{parser.prog}: error: argument --evaluate: no valid path was provided")
            sys.exit(2)
    
    args = parser.parse_args()

    data_dir = Path(config.DATA_DIR)
    preprocessed_data_output_dir = Path(config.PREPROCESSED_DATA_OUTPUT_DIR)

    if args.preprocess:
        print("Starting data preprocessing.")

        from src.preprocess import DataPreprocessor
        
        preprocessor = DataPreprocessor(data_dir=data_dir, output_dir=preprocessed_data_output_dir)
        
        preprocessor.process() 
        
        print("Data preprocessing finished.")

    if args.train:
        print("Starting training loop.")
        
        from src.training_loop import run_training
        
        run_training()
        
        print("Training finished.")

    if args.evaluate:
        print(f"Starting evaluation for checkpoint: {args.evaluate}")
        
        from src.evaluation import evaluate_model
        
        evaluate_model(model_checkpoint=args.evaluate)
        
        print("Evaluation finished.")

    if args.test:
        print("Testing...")

if __name__ == '__main__':
    main()