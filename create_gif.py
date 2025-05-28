from PIL import Image, ImageDraw
import numpy as np

# Create series of frames for a simple animation
frames = []
colors = [
    (127, 0, 255),  # Purple
    (0, 181, 226),  # Cyan
    (0, 255, 127),  # Green
    (0, 181, 226),  # Cyan back
]

width, height = 200, 200
num_frames = 20

# Create frames with simple circle animation
for i in range(num_frames):
    frame = Image.new('RGB', (width, height), (16, 24, 38))
    draw = ImageDraw.Draw(frame)
    
    # Calculate color transition
    color_idx = int(i / num_frames * len(colors))
    next_color_idx = (color_idx + 1) % len(colors)
    blend_factor = (i / num_frames * len(colors)) % 1
    
    r = int(colors[color_idx][0] * (1 - blend_factor) + colors[next_color_idx][0] * blend_factor)
    g = int(colors[color_idx][1] * (1 - blend_factor) + colors[next_color_idx][1] * blend_factor)
    b = int(colors[color_idx][2] * (1 - blend_factor) + colors[next_color_idx][2] * blend_factor)
    
    # Draw a simulated robot/face
    # Face outline
    draw.ellipse((50, 50, 150, 150), outline=(r, g, b), width=3)
    
    # Eyes
    eye_size = 10 + int(5 * np.sin(i/num_frames * 2 * np.pi))
    draw.ellipse((70 - eye_size//2, 80 - eye_size//2, 70 + eye_size//2, 80 + eye_size//2), fill=(r, g, b))
    draw.ellipse((130 - eye_size//2, 80 - eye_size//2, 130 + eye_size//2, 80 + eye_size//2), fill=(r, g, b))
    
    # Mouth
    mouth_width = 40 + int(10 * np.sin(i/num_frames * 2 * np.pi + np.pi/2))
    draw.arc((100 - mouth_width//2, 100, 100 + mouth_width//2, 130), 0, 180, fill=(r, g, b), width=3)
    
    frames.append(frame)

# Save as GIF
frames[0].save('VirtualCoach.gif', save_all=True, append_images=frames[1:], 
               duration=100, loop=0, optimize=False)

print("Created VirtualCoach.gif animation") 