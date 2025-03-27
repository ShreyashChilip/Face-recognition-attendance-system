# Face Recognition Attendance System

An automated attendance system that leverages face recognition technology to mark student attendance efficiently.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Face Recognition and Attendance](#face-recognition-and-attendance)
- [Web Interface](#web-interface)
- [Database Management](#database-management)
- [Configuration](#configuration)
- [Logging](#logging)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

The Face Recognition Attendance System is designed to automate the process of recording student attendance using facial recognition. By capturing images through a webcam, the system identifies students and logs their attendance, reducing manual effort and increasing accuracy.

## Features

- **Automated Attendance:** Recognizes faces and marks attendance without manual input.
- **Data Logging:** Stores attendance records in Excel files for easy access and management.
- **Web Interface:** Provides a user-friendly interface for interaction and monitoring.
- **Model Training:** Allows training of the recognition model with new student images.

## Project Structure

- `.devcontainer/`: Configuration files for development containers.
- `instance/`: Instance-specific files and databases.
- `templates/`: HTML templates for the web interface.
- `test-images/`: Sample images for testing the recognition system.
- `training-sh/`: Scripts related to training procedures.
- `app.py`: Main application script to run the web server.
- `count_images_of_students.py`: Counts images per student in the dataset.
- `create_db.py`: Initializes and manages the database.
- `face-detect.py`: Functions for detecting faces in images.
- `train.py`: Script to train the face recognition model.
- `webcamrecog.py`: Captures video and performs real-time face recognition.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ShreyashChilip/Face-recognition-attendance-system.git
   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd Face-recognition-attendance-system
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements-git.txt
   ```

## Usage

1. **Prepare the Dataset:** Collect and organize student images.
2. **Train the Model:** Execute the training script to update the recognition model.
3. **Run the Application:** Start the web interface.
   ```bash
   python app.py
   ```
4. **Access the Web Interface:** Open `http://localhost:5000` in a browser.

## Data Preparation

- **Collect Images:** Gather multiple images of each student and organize them.
- **Directory Structure:**
  ```
  dataset/
  ├── student_1/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  ├── student_2/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  └── ...
  ```
- **Count Images:** Use `count_images_of_students.py` to verify the dataset.

## Training the Model

- **Train with Single Thread:**
  ```bash
  python train.py
  ```
- **Train with Multi-Threading:**
  ```bash
  python train-mt.py
  ```
- **Output:** The training scripts generate `known_face_encodings.joblib` and `known_face_names.joblib`.

## Face Recognition and Attendance

- **Real-Time Recognition:** Run `webcamrecog.py` for live detection.
  ```bash
  python webcamrecog.py
  ```
- **Attendance Logging:** Recognized faces are logged into `recognized_faces.xlsx`.

## Web Interface

- **Templates:** HTML files in `templates/` define web pages.
- **Routes:** Defined in `app.py` to handle web requests.

## Database Management

- **Initialization:** `create_db.py` sets up the database schema.
- **Student Data:** `student_data.xlsx` contains student details.

## Configuration

- **Haar Cascade Classifier:** Uses `haarcascade_frontalface_default.xml`.
- **Procfile:** Defines process types for deployment.

## Logging

- **Image Uploads:** Logs stored in `image_upload_debug.log`.
- **System Logs:** Debugging information in `log.txt`.

## Dependencies

- Install Python packages:
  ```bash
  pip install -r requirements-git.txt
  ```

## License

This project is licensed under the [MIT License](LICENSE).
