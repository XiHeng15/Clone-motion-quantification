# Motion Quantification

A toolkit for capturing, analyzing, and visualizing human motion data for rehabilitation and biomechanical analysis.

## Overview

This project provides tools to:
1. Extract pose data from videos or webcam feeds using MediaPipe
2. Calculate clinically relevant kinematic metrics (joint angles, symmetry)
3. Visualize and track progress through an interactive dashboard

## Installation

1. Clone this repository
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Components

### 1. Motion Data Extraction

Two scripts are provided for capturing pose data:

#### From Video File (`motion_extract_file.py`)

Extracts pose landmarks from a pre-recorded video file.

```
python motion_extract_file.py -i <input_video_path> -o <output_csv_path>
```

Example:
```
python motion_extract_file.py -i patient_assessment.mp4 -o pose_data.csv
```

To process every video in a folder and create one matching CSV per video, pass the
folder as the input and an output folder as the output:
```
python motion_extract_file.py -i videos -o extracted_pose_data
```

This creates files like:
```
videos/session_01.mp4 -> extracted_pose_data/session_01.csv
videos/session_02.mov -> extracted_pose_data/session_02.csv
```

Useful batch options:
```
python motion_extract_file.py -i videos -o extracted_pose_data --skip-existing
python motion_extract_file.py -i videos -o extracted_pose_data --recursive
python motion_extract_file.py -i videos -o extracted_pose_data --no-annotated-video
```

#### From Webcam (`motion_extract_record.py`)

Captures live pose data from your webcam.

```
python motion_extract_record.py
```

The script will:
- Open your webcam feed
- Display pose landmarks in real-time
- Show joint angles (knees) on screen
- Save the data to a CSV file (named with timestamp) when you press 'q'

### 2. Kinematics Calculation (`kinematics_calculator.py`)

Processes the raw pose data to calculate clinically relevant metrics.

```
python kinematics_calculator.py <input_csv> --output <output_csv>
```

Example:
```
python kinematics_calculator.py pose_data.csv --output clinical_kinematics.csv
```

This script calculates:
- Joint angles (knee, hip, ankle, shoulder, elbow)
- Trunk and neck flexion angles
- Symmetry metrics between left and right sides

### 3. Rehabilitation Dashboard (`data_dashboard.py`)

An interactive GUI for visualizing and tracking rehabilitation progress.

```
python data_dashboard.py
```

Features:
- User management (add and select patients)
- Data upload (import session data)
- Visualization (trends over time, session details)
- Progress tracking with key metrics
- PDF report generation for patients

### 4. PDF Report Generation (`generate_report.py`)

Creates professional PDF reports for patients showing progress and recommendations.

```
python generate_report.py <user_id> [--data <data_file.json>] [--output <output_file.pdf>]
```

Example:
```
python generate_report.py "John Doe" --output john_doe_report.pdf
```

The report includes:
- Summary of progress across sessions
- Trend charts for key metrics
- Latest session analysis
- Personalized recommendations based on the data

## Data Flow

1. Capture raw pose data (from video or webcam)
2. Process the data to calculate clinical metrics
3. Import the processed data into the dashboard
4. Track and visualize progress over time

## Metrics

The system tracks the following biomechanical metrics:
- Range of motion for major joints (knees, hips, shoulders, elbows)
- Trunk and neck flexion angles
- Movement symmetry between left and right sides

## Screenshots

Here are some screenshots showing the application in action:

### Dashboard Overview
![Empty Dashboard](/screenshots/empty_dashboard.png)

### Creating a New User
![Make User](/screenshots/make_user.png)

### Uploading Session Data
![Upload Session](/screenshots/upload_session.png)

### Analytics - Hip Symmetry
![Hip Symmetry](/screenshots/hip_symmetry.png)

### Analytics - Knee Symmetry
![Knee Symmetry](/screenshots/knee_symmetry.png)

### Analytics - Right Knee Angle
![Right Knee Angle](/screenshots/right_knee_angle.png)

### Analytics - Trunk Flexion
![Trunk Flexion](/screenshots/trunk_flexion.png)

## Development

See CLAUDE.md for development guidelines and coding standards for this project.
