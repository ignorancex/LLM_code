import cv2
import mediapipe as mp
import numpy as np
import random
from typing import Union, List
import click

@click.command()
@click.option('--num-circles', help='Number of circles to randomly spawn', type=click.IntRange(min=1), default=5, show_default=True)
# Simulation options; change at own will
@click.option('--density', 'RHO', help='Density of the circles, will affect speeds', type=click.FloatRange(min=0.1), default=0.5, show_default=True)
@click.option('--friction', 'MU', help='Friction of the simulation: 1.0 means no friction.', type=click.FloatRange(min=0.0, max=1.0, min_open=True), default=0.995, show_default=True)
@click.option('--elastic', 'is_elastic', help='Whether to use elastic (True) or inelastic (False) collisions', type=click.BOOL, default=True, show_default=True)
def main(**kwargs):
    # print(kwargs['num_circles'], kwargs['RHO'], kwargs['MU'])
    pass

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Custom drawing specifications
landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3)  # Blue for landmarks
connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3)  # Green for connections

# Global variables for background color and display mode
background_color = (0, 0, 0)  # Black by default
display_overlay = True  # True to display the overlay of landmarks on the original image
display_connections = True  # True to display the connections between landmarks
display_tips = False  # True to display the geometric figures at the tips of each hand landmark

# Set random seed for reproducibility
random.seed(1)


# Circle object
class CircleObject:
    MU: float = 0.995  # Friction factor; 1 is no friction
    RHO: float = 0.05  # Density of the circle; mass = density * area

    def __init__(self, position: Union[list, tuple, np.ndarray], radius: int,
                 initial_velocity: Union[list, tuple, np.ndarray], screen_width: int = 1280,
                 screen_height: int = 720, color: Union[list, tuple] = (255, 0, 0)):
        # Utility function to convert input to numpy array
        def to_numpy_array(input_value, dtype=np.float32):
            if isinstance(input_value, (tuple, list)):
                return np.array(input_value, dtype=dtype)
            return input_value

        # Store the initial state
        self.initial_position = to_numpy_array(position, dtype=np.int32)  # Position vector (x, y)
        self.initial_velocity = to_numpy_array(initial_velocity)  # Velocity vector (x, y)

        # Set current position, velocity, and other attributes
        self.position = self.initial_position.copy()
        self.velocity = self.initial_velocity.copy()
        self.radius = radius
        self.color = tuple(color) if isinstance(color, np.ndarray) else color
        self.mass = self.RHO * np.pi * self.radius ** 2  # Proportional to its area: m = rho * A := density * area

        # Save the screen dimensions for spawning purposes
        self.screen_width = screen_width
        self.screen_height = screen_height

    def reset(self):
        # Reset the state of the circle to its initial state
        self.position = self.initial_position.copy()
        self.velocity = self.initial_velocity.copy()

    def update(self):
        self.velocity = self.velocity * self.MU  # Apply friction to reduce velocity
        self.position = self.position + self.velocity

        # Check for collisions with the edges of the screen and reflect velocity
        edge_bounce = 1.01  # Slight bounce factor
        if self.position[0] - self.radius <= 1e-2:
            self.velocity[0] = abs(self.velocity[0]) * edge_bounce
            self.position[0] = self.radius + 1e-2
        elif self.position[0] + self.radius >= self.screen_width - 1e-2:
            self.velocity[0] = -abs(self.velocity[0]) * edge_bounce
            self.position[0] = self.screen_width - self.radius - 1e-2

        if self.position[1] - self.radius <= 1e-2:
            self.velocity[1] = abs(self.velocity[1]) * edge_bounce
            self.position[1] = self.radius + 1e-2
        elif self.position[1] + self.radius >= self.screen_height - 1e-2:
            self.velocity[1] = -abs(self.velocity[1]) * edge_bounce
            self.position[1] = self.screen_height - self.radius - 1e-2

    def draw(self, image):
        cv2.circle(image, tuple([int(p) for p in self.position]), self.radius, self.color, -1)

    def check_collision(self, hand_position):
        distance = np.linalg.norm(self.position - hand_position)
        return 1e-2 < self.radius - distance

    # Method to check collision with another circle
    def collides_with(self, other_circle):
        distance = np.linalg.norm(self.position - other_circle.position)
        return 1e-2 < (self.radius + other_circle.radius) - distance

    # Method to handle collision response
    def handle_collision(self, other_circle, is_elastic: bool = True) -> None:
        # Save the original velocities
        original_self_velocity = self.velocity.copy()
        original_other_velocity = other_circle.velocity.copy()

        if is_elastic:
            # Simple elastic collision physics; note that if the masses are equal (i.e., equal areas), then this reduces to
            # self.velocity, other_circle.velocity = other_circle.velocity, self.velocity, but let's make it more general
            # See: https://en.wikipedia.org/wiki/Elastic_collision#Equations

            # Update velocities using the original values
            self.velocity = ((self.mass - other_circle.mass) / (
                        self.mass + other_circle.mass)) * original_self_velocity + \
                            ((2 * other_circle.mass) / (self.mass + other_circle.mass)) * original_other_velocity
            other_circle.velocity = ((2 * self.mass) / (self.mass + other_circle.mass)) * original_self_velocity + \
                                    ((other_circle.mass - self.mass) / (
                                                self.mass + other_circle.mass)) * original_other_velocity
        else:
            # Inelastic collision: https://en.wikipedia.org/wiki/Inelastic_collision#Perfectly_inelastic_collision
            self.velocity = other_circle.velocity = (self.mass * original_self_velocity + other_circle.mass * original_other_velocity) / (self.mass + other_circle.mass)

    def apply_separation_force(self, other_circle, separation_factor=0.1):
        direction = self.position - other_circle.position
        distance = np.linalg.norm(direction)
        if distance < self.radius + other_circle.radius:
            force = direction / distance * separation_factor
            self.velocity += force
            other_circle.velocity -= force

    def kinetic_energy(self):
        return 0.5 * self.mass * np.linalg.norm(self.velocity) ** 2


# Dictionary mapping modes to their respective MediaPipe solution and drawing specs
mediapipe_solutions = {
    'face': {
        'solution': mp_face_mesh.FaceMesh(),
        'connections': mp_face_mesh.FACEMESH_CONTOURS
    },
    'pose': {
        'solution': mp_pose.Pose(),
        'connections': mp_pose.POSE_CONNECTIONS
    },
    'hand': {
        'solution': mp_hands.Hands(),
        'connections': mp_hands.HAND_CONNECTIONS
    }
}


def process_landmarks(image: np.ndarray, mode: str = 'face',
                      display_overlay: bool = False, display_connections: bool = False,
                      display_tips: bool = True, circles: List[CircleObject] = None) -> np.ndarray:
    """ Process an image and display the landmarks, depending on the mode. """

    fingertip_indexes = [
        mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

    solution_info = mediapipe_solutions[mode]
    results = solution_info['solution'].process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Create a blank image for landmarks mode
    if not display_overlay:
        image = np.zeros_like(image)
        image[:] = background_color
    # Depending on the mode, landmarks are stored in different attributes
    landmarks = None
    if mode == 'face' and results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks
    elif mode == 'pose' and results.pose_landmarks:
        landmarks = [results.pose_landmarks]
    elif mode == 'hand' and results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks

    if landmarks:
        for landmark in landmarks:
            mp_drawing.draw_landmarks(
                image, landmark,
                solution_info['connections'] if display_connections else None,
                landmark_drawing_spec, connection_drawing_spec
            )
            if mode == 'hand':
                # Draw a circle around each fingertip
                if display_tips:
                    for fingertip_index in fingertip_indexes:
                        fingertip = landmark.landmark[fingertip_index]
                        x, y = int(fingertip.x * image.shape[1]), int(fingertip.y * image.shape[0])

                        # Draw a red square around the index fingertip
                        square_size = random.randint(15, 30)  # Size of the square
                        # Random color
                        color = (random.randint(0, 255),
                                 random.randint(0, 255),
                                 random.randint(0, 255))
                        cv2.rectangle(image, (x - square_size, y - square_size), (x + square_size, y + square_size),
                                      color, -1)
                # Check for collision and respond
                index_fingertip = landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                hand_pos = np.array([int(index_fingertip.x * image.shape[1]),
                                     int(index_fingertip.y * image.shape[0])], dtype=np.float32)
                for circle in circles:
                    if circle.check_collision(hand_pos):
                        circle.velocity = - (hand_pos - circle.position) * 0.3

    return image


# Initialize the virtual circle
screen_width = 1280
screen_height = 720
# circle = CircleObject(np.array([300, 300]), 100, np.array([0.1, 0.2]), screen_width, screen_height)
circles = []
num_circles = 5

def is_overlapping(new_position, new_radius, existing_circles):
    for circle in existing_circles:
        distance = np.linalg.norm(np.array(new_position) - circle.position)
        if distance + 1e-2 < (new_radius + circle.radius):
            return True  # Overlap found
    return False  # No overlap

# Function to create a new circle
def create_circle(existing_circles, screen_width, screen_height, max_attempts=100):
    for _ in range(max_attempts):
        new_radius = random.randint(50, 125)  # Or some other logic to determine the radius
        new_position = np.array([random.randint(new_radius, screen_width - new_radius),
                                 random.randint(new_radius, screen_height - new_radius)])
        if not is_overlapping(new_position, new_radius, existing_circles):
            new_circle = CircleObject(
                position=new_position,
                radius=new_radius,
                initial_velocity=np.array([random.randint(-5, 5), random.randint(-5, 5)]),
                screen_width=screen_width, screen_height=screen_height,
                color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            existing_circles.append(new_circle)
            break


for i in range(num_circles):
    create_circle(circles, screen_width, screen_height)

# Create a named window
cv2.namedWindow('MediaPipe', cv2.WINDOW_NORMAL)

# Set the window to full screen
cv2.setWindowProperty('MediaPipe', cv2.WND_PROP_FULLSCREEN, 0)

# Video capture
cap = cv2.VideoCapture(0)

# Set desired resolution to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Set desired FPS
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if the desired FPS is set correctly
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print("Actual FPS set:", actual_fps)

# Initial mode
mode = 'hand'  # Can be 'face', 'pose', 'hand'

# Create MediaPipe model objects
face_mesh = mp_face_mesh.FaceMesh()
pose = mp_pose.Pose()
hands = mp_hands.Hands()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Process according to the current mode
    image = process_landmarks(image, mode, display_overlay, display_connections, display_tips, circles)

    # Update circles and check for collisions
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            if circles[i].collides_with(circles[j]):
                circles[i].handle_collision(circles[j], False)
                circles[i].apply_separation_force(circles[j])

    # Draw circles
    for circle in circles:
        circle.update()
        circle.draw(image)

    # Resize the image to fit the screen
    image = cv2.resize(image, (1280, 768))

    # Display the appropriate image
    cv2.imshow('MediaPipe', image)

    total_kinetic_energy = sum(circle.kinetic_energy() for circle in circles)
    print("Total Kinetic Energy:", total_kinetic_energy)

    # Switch modes based on keypress
    key = cv2.waitKey(5)
    if key & 0xFF == 27:
        break
    elif key == ord('f'):
        # Press F to switch to face mode
        mode = 'face'
    elif key == ord('p'):
        # Press P to switch to pose mode
        mode = 'pose'
    elif key == ord('h'):
        # Press H to switch to hand mode
        mode = 'hand'
    elif key == ord('o'):
        # Press O to toggle overlay
        display_overlay = True
    elif key == ord('l'):
        # Press L to toggle landmarks
        display_overlay = False
    elif key == ord('c'):
        # Press C to toggle connections
        display_connections = not display_connections
    elif key == ord('t'):
        # Press T to toggle fingertip figures
        display_tips = not display_tips
    elif key == ord('r'):
        # Press R to reset circles to their initial states
        for i, circle in enumerate(circles):
            circle.reset()

# Release resources
cap.release()
cv2.destroyAllWindows()


# ==============================================================================

if __name__ == "__main__":
    main()

# ==============================================================================
