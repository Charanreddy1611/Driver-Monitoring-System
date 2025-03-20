import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved AlexNet model
model = load_model(r"C:\Users\Charan\Desktop\model_alexnet.h5")



# Define the mapping for classes to labels (for reference)
class_labels = {
    0: 'Open Eye',
    1: 'Closed Eye',
    2: 'Cigarette',
    3: 'Phone',
    4: 'Seatbelt'
}


# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame to match the model's expected input size (240x240)
    frame_resized = cv2.resize(frame, (240, 240))
    frame_normalized = frame_resized.astype("float32") / 255.0
    input_frame = np.expand_dims(frame_normalized, axis=0)

    # Get predictions from the model
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Determine driver status based on predicted class
    if predicted_class in [0, 4]:
        driver_status = "Driver is attentive"
    elif predicted_class in [2, 3]:
        driver_status = "Driver is distracted"
    elif predicted_class == 1:
        driver_status = "Driver is sleepy"
    else:
        driver_status = "Unknown state"

    # Overlay only the driver status on the original frame
    cv2.putText(frame, f"Status: {driver_status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the overlay
    cv2.imshow("Live Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


