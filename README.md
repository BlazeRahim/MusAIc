# MusAIc: emotion-aware music player (prototype)

A 2022 student project: detect the user's facial expression through the webcam and use it to choose music.

## What's in this repo
- `Emotion Detection/Model.py`: a Keras CNN trained on the FER-2013 facial-expression dataset (48×48 grayscale; angry, disgust, fear, happy, neutral, sad, surprise).
- `Emotion Detection/Emotion Detection.py`: live webcam demo. OpenCV finds faces with a Haar cascade and the CNN labels each one in real time.
- `App/`: Flutter front end (login screen).

## Run the webcam demo
```bash
cd "Emotion Detection"
pip install tensorflow opencv-python
python "Emotion Detection.py"    # press q to quit
```
It expects the trained model `model_optimal.h5` (produced by `Model.py`) in the same folder.

## Notes
This was one of my first ML projects. The repo also contains the training images, Flutter build output and a local virtual environment, which I would leave out of version control today.
