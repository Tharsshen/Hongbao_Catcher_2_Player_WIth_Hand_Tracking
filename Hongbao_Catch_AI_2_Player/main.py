import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import time

# Initialize Pygame for sound effects and visuals
pygame.init()
pygame.mixer.init()

# Sound effects
catch_sound = pygame.mixer.Sound("Resources/catch_sound.mp3")  # Replace with your sound file path
miss_sound = pygame.mixer.Sound("Resources/miss_sound.mp3")    # Replace with your sound file path

# Load images
red_envelope_img = cv2.imread("Resources/red_envelope.png", cv2.IMREAD_UNCHANGED)  # Replace with your red envelope path
firecracker_img = cv2.imread("Resources/firecracker.png", cv2.IMREAD_UNCHANGED)    # Replace with your firecracker path

# Screen settings
WIDTH, HEIGHT = 1400, 800  # Updated dimensions for a larger display

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Game variables
score_left = 0
score_right = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# Timer variables
game_duration = 60  # Total game duration in seconds
start_time = time.time()
game_over = False  # Initialize game_over flag
winner = ""  # Initialize winner variable

# Envelope and firecracker positions
red_envelope_left = {"x": random.randint(100, WIDTH // 2 - 100), "y": 0, "speed": random.randint(5, 10)}
firecracker_left = {"x": random.randint(100, WIDTH // 2 - 100), "y": 0, "speed": random.randint(5, 10)}

red_envelope_right = {"x": random.randint(WIDTH // 2 + 100, WIDTH - 100), "y": 0, "speed": random.randint(5, 10)}
firecracker_right = {"x": random.randint(WIDTH // 2 + 100, WIDTH - 100), "y": 0, "speed": random.randint(5, 10)}

# Function to overlay images with scaling
def overlay_image(frame, image, x, y, scale=0.2):
    h, w, _ = image.shape
    new_w, new_h = int(w * scale), int(h * scale)
    image_resized = cv2.resize(image, (new_w, new_h))

    x1, y1 = max(0, x - new_w // 2), max(0, y - new_h // 2)
    x2, y2 = min(frame.shape[1], x + new_w // 2), min(frame.shape[0], y + new_h // 2)

    # Adjust dimensions of the resized image if it goes out of frame
    image_cropped = image_resized[:y2 - y1, :x2 - x1]

    overlay = frame[y1:y2, x1:x2]
    alpha_mask = image_cropped[:, :, 3] / 255.0  # Assuming the image has an alpha channel

    for c in range(3):  # Apply to each channel
        overlay[..., c] = overlay[..., c] * (1 - alpha_mask) + image_cropped[:, :, c] * alpha_mask

    frame[y1:y2, x1:x2] = overlay

# Function to detect collisions
def check_collision(hand_x, hand_y, obj):
    return obj["x"] - 50 < hand_x < obj["x"] + 50 and obj["y"] - 50 < hand_y < obj["y"] + 50

# Main game loop
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # Draw the border in the middle
    cv2.line(frame, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), (255, 255, 255), 3)

    # Timer calculation
    elapsed_time = time.time() - start_time
    remaining_time = max(0, game_duration - int(elapsed_time))

    # Hand tracking
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    left_hand_x, left_hand_y = -1, -1
    right_hand_x, right_hand_y = -1, -1

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WIDTH)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * HEIGHT)

            if x < WIDTH // 2:  # Left player's side
                left_hand_x, left_hand_y = x, y
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            else:  # Right player's side
                right_hand_x, right_hand_y = x, y
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Move left red envelope and firecracker
    red_envelope_left["y"] += red_envelope_left["speed"]
    if red_envelope_left["y"] > HEIGHT:
        red_envelope_left["y"] = 0
        red_envelope_left["x"] = random.randint(100, WIDTH // 2 - 100)
        score_left -= 1  # Penalty for missing
        pygame.mixer.Sound.play(miss_sound)

    firecracker_left["y"] += firecracker_left["speed"]
    if firecracker_left["y"] > HEIGHT:
        firecracker_left["y"] = 0
        firecracker_left["x"] = random.randint(100, WIDTH // 2 - 100)

    # Check left collisions
    if check_collision(left_hand_x, left_hand_y, red_envelope_left):
        red_envelope_left["y"] = 0
        red_envelope_left["x"] = random.randint(100, WIDTH // 2 - 100)
        score_left += 10  # Award points for catching
        pygame.mixer.Sound.play(catch_sound)

    if check_collision(left_hand_x, left_hand_y, firecracker_left):
        firecracker_left["y"] = 0
        firecracker_left["x"] = random.randint(100, WIDTH // 2 - 100)
        score_left -= 10  # Penalty for hitting firecracker
        pygame.mixer.Sound.play(miss_sound)

    # Move right red envelope and firecracker
    red_envelope_right["y"] += red_envelope_right["speed"]
    if red_envelope_right["y"] > HEIGHT:
        red_envelope_right["y"] = 0
        red_envelope_right["x"] = random.randint(WIDTH // 2 + 100, WIDTH - 100)
        score_right -= 1  # Penalty for missing
        pygame.mixer.Sound.play(miss_sound)

    firecracker_right["y"] += firecracker_right["speed"]
    if firecracker_right["y"] > HEIGHT:
        firecracker_right["y"] = 0
        firecracker_right["x"] = random.randint(WIDTH // 2 + 100, WIDTH - 100)

    # Check right collisions
    if check_collision(right_hand_x, right_hand_y, red_envelope_right):
        red_envelope_right["y"] = 0
        red_envelope_right["x"] = random.randint(WIDTH // 2 + 100, WIDTH - 100)
        score_right += 10  # Award points for catching
        pygame.mixer.Sound.play(catch_sound)

    if check_collision(right_hand_x, right_hand_y, firecracker_right):
        firecracker_right["y"] = 0
        firecracker_right["x"] = random.randint(WIDTH // 2 + 100, WIDTH - 100)
        score_right -= 10  # Penalty for hitting firecracker
        pygame.mixer.Sound.play(miss_sound)

    # Draw red envelopes and firecrackers
    overlay_image(frame, red_envelope_img, red_envelope_left["x"], red_envelope_left["y"], scale=0.2)
    overlay_image(frame, firecracker_img, firecracker_left["x"], firecracker_left["y"], scale=0.2)

    overlay_image(frame, red_envelope_img, red_envelope_right["x"], red_envelope_right["y"], scale=0.2)
    overlay_image(frame, firecracker_img, firecracker_right["x"], firecracker_right["y"], scale=0.2)

    # Display scores
    cv2.putText(frame, f"Left Score: {score_left}", (50, 50), font, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Right Score: {score_right}", (WIDTH - 300, 50), font, 1, (0, 0, 255), 2)

    # Display timer
    cv2.putText(frame, f"Time Left: {remaining_time}s", (WIDTH // 2 - 150, 50), font, 1, (255, 255, 255), 2)

    # Check for game over condition
    if remaining_time == 0:
        game_over = True
        winner = "Left Player" if score_left > score_right else "Right Player" if score_right > score_left else "Tie"

    if game_over:
        cv2.putText(frame, f"Game Over! Winner: {winner}", (WIDTH // 2 - 300, HEIGHT // 2), font, 2, (0, 255, 255), 3)

    # Show the frame
    cv2.imshow("Catch the Red Envelopes - Two Player Mode", frame)

    # Wait for 'q' key press even after game over
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.quit()
