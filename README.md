# Assignment Repository

This GitHub repository contains the code and resources for an assignment with two tasks: Task 1 and Task 2. Here's an overview of what you'll find in this repository and how to use it.

## Task 1 - YOLOv4-DeepSORT

In the "yolov4-deepsort" folder, you'll find the code for Task 1, which involves object detection and tracking using YOLOv4-DeepSORT. This task generates videos with the number of detected objects displayed in the top-left corner. The resulting videos are saved in the "outputs" folder with the following names:

- `cars.avi`
- `pinkal_1.avi`
- `pinkal_2.avi`
- `pinkal_3.avi`

### Testing on a New Video

If you want to test this code on a new video, follow these simple steps:

1. Clone the GitHub Repository:

   ```bash
   # Run this command
   git clone https://github.com/Pink1619/Assignment.git
   ```

2. Set Up the Environment:

Navigate to the "yolov4-deepsort" folder and create a Python environment using the provided requirements file for GPU:

   ```bash
   # Run these commands
   cd yolov4-deepsort
   conda env create -f conda-gpu.yml
   conda activate yolov4-gpu
   pip install -r requirements-gpu.txt
   ```

3. Test on a New Video:

Use the following command to test the YOLOv4-DeepSORT model on a new video:

  ```bash
  # Run this command
  python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/new_video.mp4 --output ./outputs/out_video.avi --count
  ```

## Task 2 - Bi-Tempered Logistic Loss

For Task 2, we have replaced the categorical cross-entropy loss with the Bi-Tempered Logistic Loss, as mentioned in the provided paper. The model for the MNIST classification task is saved in this repository.

Feel free to reach out if you have any questions or need further assistance with this assignment repository. Good luck with your tasks!
