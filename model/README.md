# Models

This folder contains the pre-trained model weights and architecture definitions for our project.

## Pre-trained Weights

### File: `model_weights.pth`

This weight file (`model_weights.pth`) has been trained and optimized to work best with our current model architecture. It has been tested extensively and shows superior performance in terms of accuracy and efficiency compared to other versions.

#### Download Link
You can download the pre-trained weights from the following link:
[Download model_weights.pth](https://drive.google.com/file/d/1sLNvIH84pFDYs7IXsCWnAAH7TvbGyJtm/view?usp=sharing)

#### Usage Instructions

1. **Place the Weights:**
   - Ensure the `model_weights.pth` file is located in the `models` folder of your project directory.

2. **Loading the Weights:**
   - In your code, load the weights as follows:
     ```python
     import torch
     from Generator import Generator
     

     # Initialize the model architecture
     gen = Generator(in_channels=1)

     # Load the pre-trained weights
     gen.load_state_dict(torch.load('gen_model_30_face_25_landscape.pth'))

     # Set the model to evaluation mode
     gen.eval()
     ```

3. **Model Architecture:**
   - Ensure that the architecture of your model matches the one used to train these weights. Any discrepancies may result in errors or suboptimal performance.

4. **Performance:**
   - This model weights file has been validated to perform best on our specific dataset and tasks. If you are applying it to different datasets, consider fine-tuning the model further.

## Notes

- The `model_weights.pth` file is intended to be used as-is for our defined architecture. Modifying the architecture or using it in different contexts without proper adjustments may lead to suboptimal results.
- Regularly check for updates or new versions of the weights to ensure optimal performance and incorporate any improvements.

For more details on how to use these weights in your specific application, refer to the main project documentation or the `README.md` file in the root directory.
