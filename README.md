# Automated Attendance System

# Description
This repository contains an automated attendance system that uses facial recognition technology to identify and track attendance. The system is designed to be easy to set up and run, requiring only a few dependencies and straightforward configuration steps.

# Installation

# Step 1: Install Visual Studio with Desktop Development with C++
Make sure you have Visual Studio installed on your machine with the 'Desktop Development with C++' workload selected.

# Step 2: Install CMake
Download and install CMake from the official [CMake website](https://cmake.org/download/).

# Step 3: Clone the Repository
Clone the repository to your local machine using the following command:
```bash
git clone https://github.com/Ronin-117/Automated_attendance
```

# Step 4: Navigate to the Project Directory
Navigate to the project directory:
```bash
cd Automated_attendance
```

# Step 5: Install Python Dependencies
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

# Step 6: Add Images to the Training Folder
Add a single picture of each person's face to the `Training_images` folder. Make sure the image is clear and the face is fully visible.

# Step 7: Generate Face Encodings
Run the following script to generate face encodings from the images:
```bash
python get_encodings.py
```

# Step 8: Launch the Application
Start the attendance system by running the following command:
```bash
python app.py
```

# You can use the attendance system via a website whose web-address will be displayed in the terminal once you run the app.py . It will use the images in the `Training_images` folder to recognize faces and mark attendance.
