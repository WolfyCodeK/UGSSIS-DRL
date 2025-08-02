## Urban Green Space Sattellite Image Segmentation using Deep Reinforcement Learning

# Setup & Usage

1.  **Create and activate a virtual environment.**
 - python -m venv venv
2.  **Install dependencies:** 
 - `pip install -r requirements.txt`
 - Additionally install relevant CUDA support library. The pip install command for your specifc system can be found at https://pytorch.org/get-started/locally/
 - e.g. pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
3.  **Download FBP Dataset:** 
 - Download dataset from https://x-ytong.github.io/project/Five-Billion-Pixels.html
 - Place required subdirectories into `data/`
 - data should be directory format:

       +data
        +Annotation__color
            -GF2_PMS1__L1A0000564539-MSS1_24label.tif
            -GF2_PMS1__L1A0000575925-MSS1_24label.tif
            ...
        +Annotation__index
            -GF2_PMS1__L1A0000564539-MSS1_24label.png
            -GF2_PMS1__L1A0000575925-MSS1_24label.png
            ...
        +Coordinate_files
            -GF2_PMS1__L1A0000564539-MSS1.rpb
            -GF2_PMS1__L1A0000575925-MSS1.rpb
            ...
        +Image_8bit_NirRGB
            -GF2_PMS1__L1A0000564539-MSS1.tiff
            -GF2_PMS1__L1A0000575925-MSS1.tiff
            ...
        +Image_16bit_BGRNir
            -GF2_PMS1__L1A0000564539-MSS1.tiff
            -GF2_PMS1__L1A0000575925-MSS1.tiff
            ...
        -Five-Billion-Pixels readme (important)
        
4.  **Run Preprocessing:** 
 - `python main.py --preprocess`
5.  **Run Active Learning Training:** 
 - `python main.py --train`
7.  **Run Evaluation (Requires Training Checkpoint and Expert Labels):**
    ```bash
    python main.py --evaluate --checkpoint <path/to/model_cycle_X.pth>.
    ```
# Note:
- The preprocessing, training and evaluation steps are very long processes that can take up to 12+ hours to run on mid-tier hardware.
