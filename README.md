# Fake News Detector

## Overview
The **Fake News Detector** is a Python application that utilizes the Google Gemini API to classify news articles as real or fake. The application provides explanations for its classification results and includes functionalities for text extraction from images and speech recognition.

## Features
- **Text Extraction**: Extracts text from images using Tesseract OCR.
- **Speech Recognition**: Converts spoken words into text using the Vosk speech recognition model.
- **News Classification**: Classifies news content as real or fake using the Gemini 1.5 Flash model.

## Prerequisites
Before running the application, ensure you have the following installed:
- Python 3.7 or higher
- Tesseract OCR
- Vosk Model (download from [Vosk Models](https://alphacephei.com/vosk/models))

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/nsdivyasingh/fake-news-detection.git


2. Navigate to the project directory:
  ```bash
  cd fake-news-detection
  ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
  

## Usage
1. Run the application:
  ```bash
  python fakenews2.py
```

2. Use the GUI to upload images for text extraction, enter news content for classification, or use audio input for speech recognition.

## License
This project is licensed under the GNU General Public License v3.0 
