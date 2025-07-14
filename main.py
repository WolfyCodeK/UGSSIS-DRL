import argparse
from pathlib import Path
import src.config as config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UGS Segmentation Pipeline")
    
    parser.add_argument('--preprocess', action='store_true', help='Run data preprocessing')
    
    parser.add_argument('--train', action='store_true', help='Run the active learning training loop')
    
    parser.add_argument('--evaluate', action='store_true', help='Evaluate a trained model checkpoint')
    
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for evaluation')

    args = parser.parse_args()

    data_dir = Path(config.DATA_DIR)
    output_dir = Path(config.OUTPUT_DIR)
    test_output_dir = Path(config.TEST_OUTPUT_DIR)

    if args.preprocess:
        print("Starting data preprocessing.")

        from src.preprocess import DataPreprocessor
        
        preprocessor = DataPreprocessor(data_dir=data_dir, output_dir=output_dir)
        
        preprocessor.process() 
        
        print("Data preprocessing finished.")

    if args.train:
        print("Starting training loop.")
        
        from src.training_loop import run_training
        
        run_training()
        
        print("Training finished.")

    if args.evaluate:
        if not args.checkpoint:
            print("Checkpoint path must be provided for evaluation using --checkpoint")
            
        else:
            print(f"Starting evaluation for checkpoint: {args.checkpoint}")
            
            from src.evaluation import evaluate_model
            
            evaluate_model(model_checkpoint=args.checkpoint)
            
            print("Evaluation finished.")