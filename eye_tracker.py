import cv2
import pyautogui
import time

# Load Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize variables
blink_start_time = None  # To track blinking duration
exit_start_time = None   # To track exit gesture
screen_w, screen_h = pyautogui.size()
eye_selected = None

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip frame horizontally
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
        for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # Use the first two detected eyes
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2
            cv2.circle(frame, (eye_center_x, eye_center_y), 10, (0, 255, 0), 2)

            # Assign left or right eye
            if i == 0:  # Assume first eye is left, second is right
                eye_selected = 'left'
                mouse_x = int(screen_w / frame.shape[1] * eye_center_x)
                mouse_y = int(screen_h / frame.shape[0] * eye_center_y)
                pyautogui.moveTo(mouse_x, mouse_y)
            elif i == 1:  # Right eye for clicking
                eye_selected = 'right'

        # Detect if both eyes are closed
        if len(eyes) == 0:
            if exit_start_time is None:
                exit_start_time = time.time()  # Start timing for exit gesture
            elif time.time() - exit_start_time >= 3:  # Both eyes closed for 3 seconds
                print("Exiting program...")
                cap.release()
                cv2.destroyAllWindows()
                exit()
        else:
            exit_start_time = None  # Reset timer if eyes are detected

        # Blink detection for clicking
        if len(eyes) == 1 and eye_selected == 'right':
            if blink_start_time is None:
                blink_start_time = time.time()
            elif time.time() - blink_start_time >= 1:  # Right eye blink detected for 1 second
                pyautogui.click()
                print("Mouse clicked!")
                blink_start_time = None
        else:
            blink_start_time = None  # Reset blink timer if no blink detected

    # Show the frame
    cv2.imshow('Eye Tracker', frame)

    # Break if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
