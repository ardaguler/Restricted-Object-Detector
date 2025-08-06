# Restricted Object Detector

This is an object detection project developed to identify and flag sensitive content such as knives, guns, and cigarettes in images. The Roboflow platform was used for dataset management and model training.

<p align="center">
  <a href="https://roboflow.com" target="_blank">
    <img src="https://vectorseek.com/wp-content/uploads/2025/08/Roboflow-Ai-Logo-PNG-SVG-Vector.png" height="50">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://ultralytics.com" target="_blank">
    <img src="https://cdn.prod.website-files.com/680a070c3b99253410dd3dcf/680a070c3b99253410dd3e88_UltralyticsYOLO_full_blue.svg" height="50">
  </a>
</p>

## Features

* **Local Detection:** Performs object detection locally using a trained YOLOv8 model (`.pt` weights file) without needing an internet connection.
* **Visualization:** Displays the detected objects and their confidence scores directly on the original image.

## Technologies Used

* **Python 3**
* **Ultralytics (YOLOv8):** For loading the model and running inference.
* **OpenCV:** For image processing and visualization.
* **Roboflow:** For dataset management and model training.

## Setup and Usage

To run this project on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/ardaguler/Restricted-Object-Detector.git](https://github.com/ardaguler/Restricted-Object-Detector.git)
    cd Restricted-Object-Detector
    ```

2.  **Install Dependencies:**
    Make sure you have a Python environment set up. Then, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the Model (Important):**
    This repository does not include the model's weight file (`.pt`) due to its large size. You need to place your trained `weights.pt` file inside the `Models` folder.

4.  **Run the Script:**
    Update the `IMAGE_PATH` variable in the Python script to point to your desired image. Then, run the script:
    ```bash
    python your_script_name.py
    ```
    *(Not: `your_script_name.py` kısmını kendi Python dosyanın adıyla değiştir, örn: `with_weights.py`)*

## Future Enhancements

-   [ ] Adding real-time detection capability from a video feed or webcam.
-   [ ] Logging the detection results to a text or CSV file.
-   [ ] Creating a simple user interface (UI) for easier use.