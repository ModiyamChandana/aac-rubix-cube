"""
3D Rubik's Cube Environment for Reinforcement Learning - FIXED VERSION
"""

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import gymnasium as gym
from gymnasium import spaces
import random

# Define the colors of the cube
COLORS = {
    'white': (1, 1, 1),
    'yellow': (1, 1, 0),
    'blue': (0, 0, 1),
    'green': (0, 1, 0),
    'red': (1, 0, 0),
    'orange': (1, 0.5, 0)
}

# Mapping of face indices to colors
FACE_COLORS = {
    0: COLORS['white'],   # Up
    1: COLORS['yellow'],  # Down
    2: COLORS['blue'],    # Front
    3: COLORS['green'],   # Back
    4: COLORS['orange'],  # Right
    5: COLORS['red']      # Left
}

class RubiksCube:
    """
    A 3×3×3 Rubik's Cube representation
    """
    def _init_(self):
        # Each face is a 3×3 array of colors
        # 0: Up, 1: Down, 2: Front, 3: Back, 4: Right, 5: Left
        self.state = np.zeros((6, 3, 3), dtype=int)
        
        # Initialize a solved cube
        for i in range(6):
            self.state[i, :, :] = i
            
    def reset(self):
        """Reset the cube to a solved state"""
        for i in range(6):
            self.state[i, :, :] = i
            
    def get_state(self):
        """Return a flattened representation of the cube state"""
        return self.state.flatten()
    
    def is_solved(self):
        """Check if the cube is solved"""
        for i in range(6):
            if not np.all(self.state[i] == i):
                return False
        return True
    
    def rotate_face(self, face, clockwise=True):
        """Rotate a face clockwise or counterclockwise"""
        # Create a copy of the face
        face_copy = self.state[face].copy()
        
        if clockwise:
            # Rotate 90 degrees clockwise
            for i in range(3):
                for j in range(3):
                    self.state[face, i, j] = face_copy[2-j, i]
        else:
            # Rotate 90 degrees counterclockwise
            for i in range(3):
                for j in range(3):
                    self.state[face, i, j] = face_copy[j, 2-i]
            
    def update_adjacent_edges(self, face, clockwise=True):
        """Update the adjacent edges when a face is rotated - FIXED VERSION"""
        
        if face == 0:  # U (Up) face
            # Save the edges that will be moved
            temp = self.state[2, 0, :].copy()  # Front top row
            if clockwise:
                # Clockwise: Front <- Right <- Back <- Left <- Front
                self.state[2, 0, :] = self.state[4, 0, :]  # Front <- Right
                self.state[4, 0, :] = self.state[3, 0, :]  # Right <- Back  
                self.state[3, 0, :] = self.state[5, 0, :]  # Back <- Left
                self.state[5, 0, :] = temp                 # Left <- Front
            else:
                # Counter-clockwise: Front <- Left <- Back <- Right <- Front
                self.state[2, 0, :] = self.state[5, 0, :]  # Front <- Left
                self.state[5, 0, :] = self.state[3, 0, :]  # Left <- Back
                self.state[3, 0, :] = self.state[4, 0, :]  # Back <- Right
                self.state[4, 0, :] = temp                 # Right <- Front
                
        elif face == 1:  # D (Down) face
            # Save the edges that will be moved
            temp = self.state[2, 2, :].copy()  # Front bottom row
            if clockwise:
                # Clockwise: Front <- Left <- Back <- Right <- Front
                self.state[2, 2, :] = self.state[5, 2, :]  # Front <- Left
                self.state[5, 2, :] = self.state[3, 2, :]  # Left <- Back
                self.state[3, 2, :] = self.state[4, 2, :]  # Back <- Right
                self.state[4, 2, :] = temp                 # Right <- Front
            else:
                # Counter-clockwise: Front <- Right <- Back <- Left <- Front
                self.state[2, 2, :] = self.state[4, 2, :]  # Front <- Right
                self.state[4, 2, :] = self.state[3, 2, :]  # Right <- Back
                self.state[3, 2, :] = self.state[5, 2, :]  # Back <- Left
                self.state[5, 2, :] = temp                 # Left <- Front
                
        elif face == 2:  # F (Front) face
            # Save the edges that will be moved
            temp = self.state[0, 2, :].copy()  # Up bottom row
            if clockwise:
                # Clockwise cycle
                self.state[0, 2, :] = self.state[5, :, 2][::-1]  # Up <- Left (column, reversed)
                self.state[5, :, 2] = self.state[1, 0, :]        # Left <- Down (row)
                self.state[1, 0, :] = self.state[4, :, 0][::-1]  # Down <- Right (column, reversed)
                self.state[4, :, 0] = temp                       # Right <- Up (row)
            else:
                # Counter-clockwise cycle
                self.state[0, 2, :] = self.state[4, :, 0]        # Up <- Right (column)
                self.state[4, :, 0] = self.state[1, 0, :][::-1]  # Right <- Down (row, reversed)
                self.state[1, 0, :] = self.state[5, :, 2]        # Down <- Left (column)
                self.state[5, :, 2] = temp[::-1]                 # Left <- Up (row, reversed)
                
        elif face == 3:  # B (Back) face
            # Save the edges that will be moved
            temp = self.state[0, 0, :].copy()  # Up top row
            if clockwise:
                # Clockwise cycle
                self.state[0, 0, :] = self.state[4, :, 2]        # Up <- Right (column)
                self.state[4, :, 2] = self.state[1, 2, :][::-1]  # Right <- Down (row, reversed)
                self.state[1, 2, :] = self.state[5, :, 0]        # Down <- Left (column)
                self.state[5, :, 0] = temp[::-1]                 # Left <- Up (row, reversed)
            else:
                # Counter-clockwise cycle
                self.state[0, 0, :] = self.state[5, :, 0][::-1]  # Up <- Left (column, reversed)
                self.state[5, :, 0] = self.state[1, 2, :]        # Left <- Down (row)
                self.state[1, 2, :] = self.state[4, :, 2][::-1]  # Down <- Right (column, reversed)
                self.state[4, :, 2] = temp                       # Right <- Up (row)
                
        elif face == 4:  # R (Right) face
            # Save the edges that will be moved
            temp = self.state[0, :, 2].copy()  # Up right column
            if clockwise:
                # Clockwise cycle
                self.state[0, :, 2] = self.state[2, :, 2]        # Up <- Front (column)
                self.state[2, :, 2] = self.state[1, :, 2]        # Front <- Down (column)
                self.state[1, :, 2] = self.state[3, :, 0][::-1]  # Down <- Back (column, reversed)
                self.state[3, :, 0] = temp[::-1]                 # Back <- Up (column, reversed)
            else:
                # Counter-clockwise cycle
                self.state[0, :, 2] = self.state[3, :, 0][::-1]  # Up <- Back (column, reversed)
                self.state[3, :, 0] = self.state[1, :, 2][::-1]  # Back <- Down (column, reversed)
                self.state[1, :, 2] = self.state[2, :, 2]        # Down <- Front (column)
                self.state[2, :, 2] = temp                       # Front <- Up (column)
                
        elif face == 5:  # L (Left) face
            # Save the edges that will be moved
            temp = self.state[0, :, 0].copy()  # Up left column
            if clockwise:
                # Clockwise cycle
                self.state[0, :, 0] = self.state[3, :, 2][::-1]  # Up <- Back (column, reversed)
                self.state[3, :, 2] = self.state[1, :, 0][::-1]  # Back <- Down (column, reversed)
                self.state[1, :, 0] = self.state[2, :, 0]        # Down <- Front (column)
                self.state[2, :, 0] = temp                       # Front <- Up (column)
            else:
                # Counter-clockwise cycle
                self.state[0, :, 0] = self.state[2, :, 0]        # Up <- Front (column)
                self.state[2, :, 0] = self.state[1, :, 0]        # Front <- Down (column)
                self.state[1, :, 0] = self.state[3, :, 2][::-1]  # Down <- Back (column, reversed)
                self.state[3, :, 2] = temp[::-1]                 # Back <- Up (column, reversed)
                
    def apply_move(self, move):
        """Apply a move to the cube"""
        face, clockwise = move
        # First rotate the face itself
        self.rotate_face(face, clockwise)
        # Then update the adjacent edges
        self.update_adjacent_edges(face, clockwise)
        
    def apply_moves(self, moves):
        """Apply a sequence of moves to the cube"""
        for move in moves:
            self.apply_move(move)
            
    def scramble(self, num_moves=20):
        """Scramble the cube with random moves"""
        moves = []
        for _ in range(num_moves):
            face = random.randint(0, 5)
            clockwise = random.choice([True, False])
            moves.append((face, clockwise))
        
        self.apply_moves(moves)
        return moves


class RubiksCubeEnv(gym.Env):
    """
    Gymnasium environment for Rubik's Cube
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def _init_(self, max_steps=50, render_mode=None):
        super(RubiksCubeEnv, self)._init_()
        
        # Action space: (face, direction)
        # 6 faces, 2 directions (clockwise/counterclockwise)
        self.action_space = spaces.Discrete(12)
        
        # Observation space: 6 faces, 3×3 grid, 6 colors
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(6, 3, 3), dtype=np.int32
        )
        
        self.cube = RubiksCube()
        self.max_steps = max_steps
        self.steps = 0
        self.scramble_moves = None
        self.render_mode = render_mode
        self.renderer = None
        
    def _action_to_move(self, action):
        """Convert action index to (face, clockwise) tuple"""
        face = action // 2
        clockwise = action % 2 == 0
        return (face, clockwise)
    
    def reset(self, seed=None, options=None):
        """Reset the environment with a scrambled cube"""
        super().reset(seed=seed)
        self.cube.reset()
        scramble_steps = 20
        if options and 'scramble_steps' in options:
            scramble_steps = options['scramble_steps']
        self.scramble_moves = self.cube.scramble(scramble_steps)
        self.steps = 0
        
        if self.render_mode == 'human' and self.renderer is None:
            self.renderer = RubiksCubeRenderer(self)
            
        return self.cube.state, {}
    
    def step(self, action):
        """Take a step in the environment"""
        self.steps += 1
        
        # Apply the move
        move = self._action_to_move(action)
        self.cube.apply_move(move)
        
        # Check if the cube is solved
        solved = self.cube.is_solved()
        
        # Calculate reward
        if solved:
            reward = 10.0  # Higher reward for solving the cube
        else:
            # Use a negative reward proportional to the number of misplaced cubies
            num_misplaced = 0
            for i in range(6):
                num_misplaced += np.sum(self.cube.state[i] != i)
            
            # Normalize the reward (-1 to 0)
            reward = -num_misplaced / 54.0
        
        # Check if maximum steps reached
        terminated = solved or self.steps >= self.max_steps
        truncated = False
        
        # Render if human mode
        if self.render_mode == 'human' and self.renderer:
            self.renderer.render()
        
        return self.cube.state, reward, terminated, truncated, {}
    
    def render(self):
        """Render the cube"""
        if self.render_mode == 'human':
            if self.renderer is None:
                self.renderer = RubiksCubeRenderer(self)
            return self.renderer.render()
        return self.cube.state
    
    def close(self):
        """Close the environment"""
        if self.renderer:
            self.renderer.close()
            self.renderer = None
    

class RubiksCubeRenderer:
    """
    Renderer for the Rubik's Cube using PyGame and OpenGL
    """
    def _init_(self, env):
        self.env = env
        self.size = (800, 600)
        self.bg_color = (1, 1, 1)  # White background
        self.initialized = False
        
        # Camera settings
        self.distance = 10
        self.rotation_x = 30
        self.rotation_y = 45
        
        # Mouse interaction
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
    def init_display(self):
        """Initialize the display"""
        pygame.init()
        pygame.display.set_mode(self.size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Rubik's Cube RL - Use Mouse to Rotate View")
        
        # Set up OpenGL
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up the projection matrix
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.size[0] / self.size[1]), 0.1, 50.0)
        
        # Set up the model view matrix
        glMatrixMode(GL_MODELVIEW)
        
        # Set up the light
        glLight(GL_LIGHT0, GL_POSITION, (5, 5, 5, 1))
        glLight(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
        glLight(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
        
        self.initialized = True
        
    def draw_cube(self):
        """Draw the Rubik's Cube"""
        cube_state = self.env.cube.state
        
        # Draw each cubie
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    self.draw_cubie(x, y, z, cube_state)
    
    def draw_cubie(self, x, y, z, cube_state):
        """Draw a single cubie at the given position"""
        # Skip the center cubie
        if x == 0 and y == 0 and z == 0:
            return
        
        # Save the current matrix
        glPushMatrix()
        
        # Position the cubie
        glTranslatef(x * 1.05, y * 1.05, z * 1.05)
        
        # Draw each face of the cubie - CORRECTED MAPPING
        sides = [
            # Up face (Y = +1)
            ([0, 1, 0], 0, x+1, 1-z),
            # Down face (Y = -1)  
            ([0, -1, 0], 1, x+1, z+1),
            # Front face (Z = +1)
            ([0, 0, 1], 2, x+1, 1-y),
            # Back face (Z = -1)
            ([0, 0, -1], 3, 1-x, 1-y),
            # Right face (X = +1)
            ([1, 0, 0], 4, z+1, 1-y),
            # Left face (X = -1)
            ([-1, 0, 0], 5, 1-z, 1-y)
        ]
        
        for normal, face_idx, i, j in sides:
            # Only draw the face if it's on the outside of the cube
            if (normal[0] != 0 and abs(x) == 1) or \
               (normal[1] != 0 and abs(y) == 1) or \
               (normal[2] != 0 and abs(z) == 1):
                
                # Clamp indices to valid range
                i = max(0, min(2, int(i)))
                j = max(0, min(2, int(j)))
                
                # Get the color of this face
                color_idx = cube_state[face_idx, j, i]
                color = FACE_COLORS[color_idx]
                
                # Draw the face
                self.draw_face(normal, color)
        
        # Restore the matrix
        glPopMatrix()
    
    def draw_face(self, normal, color):
        """Draw a single face of a cubie"""
        # Set the color
        glColor3fv(color)
        
        # Calculate the vertices
        size = 0.5
        if normal[0] != 0:  # X-axis
            vertices = [
                (normal[0] * size, -size, -size),
                (normal[0] * size, size, -size),
                (normal[0] * size, size, size),
                (normal[0] * size, -size, size)
            ]
        elif normal[1] != 0:  # Y-axis
            vertices = [
                (-size, normal[1] * size, -size),
                (size, normal[1] * size, -size),
                (size, normal[1] * size, size),
                (-size, normal[1] * size, size)
            ]
        else:  # Z-axis
            vertices = [
                (-size, -size, normal[2] * size),
                (size, -size, normal[2] * size),
                (size, size, normal[2] * size),
                (-size, size, normal[2] * size)
            ]
        
        # Draw the face with a black outline
        glBegin(GL_QUADS)
        glNormal3fv(normal)
        for vertex in vertices:
            glVertex3fv(vertex)
        glEnd()
        
        # Draw black outline
        glColor3f(0, 0, 0)
        glLineWidth(2)
        glBegin(GL_LINE_LOOP)
        for vertex in vertices:
            glVertex3fv(vertex)
        glEnd()
    
    def render(self):
        """Render the cube"""
        if not self.initialized:
            self.init_display()
        
        # Clear the screen with white background
        glClearColor(*self.bg_color, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up the camera
        glLoadIdentity()
        gluLookAt(0, 0, self.distance, 0, 0, 0, 0, 1, 0)
        
        # Apply rotations
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        
        # Draw the cube
        self.draw_cube()
        
        # Update the display
        pygame.display.flip()
        
        # Handle events
        return self.handle_events()
        
    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
                
            # Mouse events for cube rotation
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.mouse_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    self.mouse_dragging = False
                    
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    mouse_pos = pygame.mouse.get_pos()
                    dx = mouse_pos[0] - self.last_mouse_pos[0]
                    dy = mouse_pos[1] - self.last_mouse_pos[1]
                    
                    # Update rotation based on mouse movement
                    self.rotation_y += dx * 0.5
                    self.rotation_x += dy * 0.5
                    
                    # Keep rotation within reasonable bounds
                    self.rotation_x = max(-90, min(90, self.rotation_x))
                    
                    self.last_mouse_pos = mouse_pos
                    
            # Mouse wheel for zooming
            elif event.type == pygame.MOUSEWHEEL:
                self.distance -= event.y * 0.5
                self.distance = max(5, min(20, self.distance))
            
            elif event.type == pygame.KEYDOWN:
                # Arrow keys for cube rotation (alternative to mouse)
                if event.key == pygame.K_LEFT:
                    self.rotation_y -= 10
                elif event.key == pygame.K_RIGHT:
                    self.rotation_y += 10
                elif event.key == pygame.K_UP:
                    self.rotation_x -= 10
                elif event.key == pygame.K_DOWN:
                    self.rotation_x += 10
                    
                # Zoom with + and - keys
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.distance -= 1
                    self.distance = max(5, self.distance)
                elif event.key == pygame.K_MINUS:
                    self.distance += 1
                    self.distance = min(20, self.distance)
                    
                # Reset view
                elif event.key == pygame.K_r:
                    self.rotation_x = 30
                    self.rotation_y = 45
                    self.distance = 10
                    
                # Preset views
                elif event.key == pygame.K_1:  # Front view
                    self.rotation_x = 0
                    self.rotation_y = 0
                elif event.key == pygame.K_2:  # Back view
                    self.rotation_x = 0
                    self.rotation_y = 180
                elif event.key == pygame.K_3:  # Left view
                    self.rotation_x = 0
                    self.rotation_y = -90
                elif event.key == pygame.K_4:  # Right view
                    self.rotation_x = 0
                    self.rotation_y = 90
                elif event.key == pygame.K_5:  # Top view
                    self.rotation_x = -90
                    self.rotation_y = 0
                elif event.key == pygame.K_6:  # Bottom view
                    self.rotation_x = 90
                    self.rotation_y = 0
                    
                # Manual cube rotations (for testing)
                elif event.key == pygame.K_u:  # Up face
                    self.env.cube.apply_move((0, True))
                elif event.key == pygame.K_d:  # Down face
                    self.env.cube.apply_move((1, True))
                elif event.key == pygame.K_f:  # Front face
                    self.env.cube.apply_move((2, True))
                elif event.key == pygame.K_b:  # Back face
                    self.env.cube.apply_move((3, True))
                elif event.key == pygame.K_l:  # Left face
                    self.env.cube.apply_move((5, True))
                elif event.key == pygame.K_g:  # Right face
                    self.env.cube.apply_move((4, True))
                elif event.key == pygame.K_s:  # Scramble
                    self.env.cube.scramble(10)
                elif event.key == pygame.K_c:  # Reset to solved
                    self.env.cube.reset()
                    
                # Show help
                elif event.key == pygame.K_h:
                    self.print_help()
        
        return True
    
    def print_help(self):
        """Print help information to console"""
        print("\n=== Rubik's Cube Controls ===")
        print("Mouse:")
        print("  Left Click + Drag: Rotate view")
        print("  Mouse Wheel: Zoom in/out")
        print()
        print("Keyboard:")
        print("  Arrow Keys: Rotate view")
        print("  +/- Keys: Zoom in/out")
        print("  R: Reset view")
        print("  H: Show this help")
        print()
        print("Preset Views:")
        print("  1: Front view")
        print("  2: Back view") 
        print("  3: Left view")
        print("  4: Right view")
        print("  5: Top view")
        print("  6: Bottom view")
        print()
        print("Cube Operations:")
        print("  U: Rotate Up face")
        print("  D: Rotate Down face")
        print("  F: Rotate Front face")
        print("  B: Rotate Back face")
        print("  L: Rotate Left face")
        print("  G: Rotate Right face")
        print("  S: Scramble cube")
        print("  C: Reset to solved state")
        print("========================\n")
    
    def close(self):
        """Close the display"""
        pygame.quit()


# Example usage
if _name_ == "_main_":
    # Create the environment
    env = RubiksCubeEnv(render_mode='human')
    
    # Reset the environment (scramble the cube)
    state, _ = env.reset(options={'scramble_steps': 5})
    
    print("=== Rubik's Cube Visualization - FIXED VERSION ===")
    print("Controls:")
    print("  Mouse: Left click + drag to rotate view")
    print("  Mouse wheel: Zoom in/out")
    print("  Arrow keys: Rotate view")
    print("  Numbers 1-6: Preset views")
    print("  R: Reset view")
    print("  H: Show detailed help")
    print("  S: Scramble cube")
    print("  C: Reset to solved state")
    print("===================================\n")
    
    # Main loop - just for visualization
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Render at 30 FPS
        clock.tick(30)
        
        # Render the cube (this also handles events)
        running = env.render()
        
    env.close()