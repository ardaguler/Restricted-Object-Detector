# Restricted Object Detector

This is an object detection project developed to identify and flag sensitive content such as knives, guns, and cigarettes in images. The Roboflow platform was used for dataset management and model training.

<p align="center">
  <a href="https://roboflow.com" target="_blank">
    <img src="https://techcrunch.com/wp-content/uploads/2021/01/roboflow_raccoon_full.png" height="50">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://ultralytics.com" target="_blank">
    <img src="https://cdn.prod.website-files.com/680a070c3b99253410dd3dcf/680a070c3b99253410dd3e88_UltralyticsYOLO_full_blue.svg" height="50">
  </a>
</p>

## Features

- **Local Detection:** Performs object detection locally using a trained YOLOv8 model (`.pt` weights file) without needing an internet connection.
- **Visualization:** Displays the detected objects and their confidence scores directly on the original image.

---

## Technologies Used

This project is built with the following core technologies:

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/YOLOv8-8D44AD?style=for-the-badge&logoColor=white" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Roboflow-000000?style=for-the-badge&logo=roboflow&logoColor=white" alt="Roboflow">
</p>

---

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
    Update the `IMAGE_PATH` variable in `with_weights.py` to point to your desired image. Then, run the script:
    ```bash
    python with_weights.py
    ```
---

## Demonstration

Here is an example of the model detecting a 'gun' in a test image. The bounding box, class label, and confidence score are displayed on the output image.

![Detection Example](https://github.com/user-attachments/assets/a6ef177f-d69b-443e-bd1b-4a284d8572d2)

*An example of the model successfully identifying a firearm with a high confidence score.*

---

## Future Enhancements

- [ ] Adding real-time detection capability from a video feed or webcam.
- [ ] Logging the detection results to a text or CSV file.
- [ ] Creating a simple user interface (UI) for easier use.