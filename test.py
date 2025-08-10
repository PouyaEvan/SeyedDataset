import os
import argparse
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import logging
from typing import Tuple, List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SHAPE_CLASSES = {'square': 0, 'circle': 1, 'triangle': 2}
MIN_SHAPES_PER_IMAGE = 1
MAX_SHAPES_PER_IMAGE = 3  # Reduced for CanSat realism
MIN_SHAPE_SIZE = 30  # Smaller for more realistic CanSat scale
MAX_SHAPE_SIZE = 150  # Adjusted for CanSat perspective
SHADOW_OFFSET = (3, 3)  # Smaller shadow for higher altitude perspective
SHADOW_BLUR_RADIUS = 10  # Less blur for CanSat altitude
SHADOW_OPACITY = 0.2  # Lighter shadows for daylight aerial view
SHAPE_OPACITY = 0.88  # Slightly more transparent for realism
TEXTURE_OPACITY = 0.12  # Subtle texture
PERSPECTIVE_DISTORTION = 0.2  # Less distortion for CanSat perspective

class SyntheticDataGenerator:
    
    def __init__(self, input_dir: str, output_dir: str, num_images: int):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_images = num_images
        
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        self.background_images = self._load_background_images()
        if not self.background_images:
            raise ValueError(f"No valid images found in {input_dir}")
        
        logger.info(f"Loaded {len(self.background_images)} background images")
    
    def _load_background_images(self) -> List[Path]:
        """Load all valid image files from input directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = []
        for file_path in self.input_dir.iterdir():
            if file_path.suffix.lower() in valid_extensions:
                images.append(file_path)
        return images
    
    def generate_dataset(self):
        """Generate the complete synthetic dataset."""
        logger.info(f"Starting generation of {self.num_images} synthetic images...")
        
        # Create classes.txt file for YOLO
        classes_file = self.output_dir / 'classes.txt'
        with open(classes_file, 'w') as f:
            for shape_name, class_id in sorted(SHAPE_CLASSES.items(), key=lambda x: x[1]):
                f.write(f"{shape_name}\n")
        logger.info(f"Created classes file: {classes_file}")
        
        for i in range(self.num_images):
            bg_path = random.choice(self.background_images)
            background = cv2.imread(str(bg_path))
            if background is None:
                logger.warning(f"Failed to load {bg_path}, skipping...")
                continue
            
            # Generate synthetic image with shapes
            result_image, annotations = self._generate_single_image(background)
            
            # Save image and annotations
            output_name = f"cansat_synthetic_{i:06d}"
            self._save_results(result_image, annotations, output_name)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{self.num_images} images")
        
        logger.info("CanSat synthetic dataset generation complete!")
    
    def _generate_single_image(self, background: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Generate a single synthetic image with multiple shapes."""
        height, width = background.shape[:2]
        result = background.copy()
        annotations = []
        
        # Check if image is large enough for shape generation
        min_dimension = min(height, width)
        if min_dimension < 300:  # Skip very small images
            logger.warning(f"Image too small ({width}x{height}), skipping shape generation")
            return result, annotations
        
        # Determine number of shapes to add
        num_shapes = random.randint(MIN_SHAPES_PER_IMAGE, MAX_SHAPES_PER_IMAGE)
        
        for _ in range(num_shapes):
            # Random shape parameters
            shape_type = random.choice(['square', 'circle', 'triangle'])
            # Adjust shape size based on image dimensions
            max_size = min(MAX_SHAPE_SIZE, min_dimension // 4)
            size = random.randint(MIN_SHAPE_SIZE, max_size)
            color = self._generate_random_color()
            
            # Random position (ensuring shape fits within image)
            margin = size + 50
            if margin * 2 >= width or margin * 2 >= height:
                continue  # Skip if image is too small for this shape
                
            x = random.randint(margin, width - margin)
            y = random.randint(margin, height - margin)
            
            # Create and apply shape
            shape_mask, shape_image = self._create_shape(shape_type, size, color)
            
            # Apply perspective transformation
            warped_shape, warped_mask, transform_matrix = self._apply_perspective_warp(
                shape_image, shape_mask, size
            )
            
            # Create shadow
            shadow = self._create_shadow(warped_mask)
            
            # Apply texture
            textured_shape = self._apply_texture(warped_shape, warped_mask)
            
            # Calculate placement position
            placement_x = x - size // 2
            placement_y = y - size // 2
            
            # Apply shadow first
            result = self._blend_shadow(result, shadow, placement_x, placement_y)
            
            # Blend shape with background
            result = self._blend_shape(result, textured_shape, warped_mask, 
                                     placement_x, placement_y)
            
            # Calculate bounding box for YOLO annotation
            bbox = self._calculate_bounding_box(warped_mask, placement_x, placement_y)
            if bbox:
                annotation = {
                    'class_id': SHAPE_CLASSES[shape_type],
                    'bbox': bbox,
                    'image_width': width,
                    'image_height': height
                }
                annotations.append(annotation)
        
        return result, annotations
    
    def _generate_random_color(self) -> Tuple[int, int, int]:
        """Generate a random color with some constraints for realism."""
        # Generate colors that are not too dark or too bright
        hue = random.randint(0, 360)
        saturation = random.uniform(0.3, 0.8)
        value = random.uniform(0.4, 0.9)
        
        # Convert HSV to RGB
        c = value * saturation
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = value - c
        
        if 0 <= hue < 60:
            r, g, b = c, x, 0
        elif 60 <= hue < 120:
            r, g, b = x, c, 0
        elif 120 <= hue < 180:
            r, g, b = 0, c, x
        elif 180 <= hue < 240:
            r, g, b = 0, x, c
        elif 240 <= hue < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
    
    def _create_shape(self, shape_type: str, size: int, color: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Create a shape with transparent background."""
        # Create larger canvas for perspective transform
        canvas_size = size * 2
        shape_image = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
        
        # Create shape in center of canvas
        center = canvas_size // 2
        
        if shape_type == 'square':
            half_size = size // 2
            cv2.rectangle(shape_image, 
                         (center - half_size, center - half_size),
                         (center + half_size, center + half_size),
                         (*color, 255), -1)
        elif shape_type == 'circle':
            cv2.circle(shape_image, (center, center), size // 2, (*color, 255), -1)
        else:  # triangle
            half_size = size // 2
            triangle_points = np.array([
                [center, center - half_size],  # top point
                [center - half_size, center + half_size],  # bottom left
                [center + half_size, center + half_size]   # bottom right
            ], np.int32)
            cv2.fillPoly(shape_image, [triangle_points], (*color, 255))
        
        # Extract alpha channel as mask
        mask = shape_image[:, :, 3]
        
        # Convert to BGR for OpenCV
        shape_bgr = cv2.cvtColor(shape_image[:, :, :3], cv2.COLOR_RGB2BGR)
        
        return mask, shape_bgr
    
    def _apply_perspective_warp(self, shape_image: np.ndarray, mask: np.ndarray, 
                               original_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply perspective transformation to simulate ground-level view."""
        h, w = shape_image.shape[:2]
        
        # Define source points (corners of the shape's bounding box)
        center = w // 2
        half_size = original_size // 2
        src_points = np.float32([
            [center - half_size, center - half_size],
            [center + half_size, center - half_size],
            [center + half_size, center + half_size],
            [center - half_size, center + half_size]
        ])
        
        # Create destination points with perspective distortion
        # Top edge compressed more than bottom edge for aerial view
        perspective_factor = PERSPECTIVE_DISTORTION * random.uniform(0.5, 1.0)
        top_compression = random.uniform(0.7, 0.9)
        
        # Add some rotation
        angle = random.uniform(-30, 30)
        
        dst_points = np.float32([
            [center - half_size * top_compression + random.randint(-10, 10), 
             center - half_size + random.randint(-10, 10)],
            [center + half_size * top_compression + random.randint(-10, 10), 
             center - half_size + random.randint(-10, 10)],
            [center + half_size + random.randint(-10, 10), 
             center + half_size + random.randint(-10, 10)],
            [center - half_size + random.randint(-10, 10), 
             center + half_size + random.randint(-10, 10)]
        ])
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        warped_shape = cv2.warpPerspective(shape_image, matrix, (w, h))
        warped_mask = cv2.warpPerspective(mask, matrix, (w, h))
        
        return warped_shape, warped_mask, matrix
    
    def _create_shadow(self, mask: np.ndarray) -> np.ndarray:
        """Create a realistic drop shadow from the shape mask."""
        # Create shadow from mask
        shadow = mask.copy()
        
        # Offset shadow
        M = np.float32([[1, 0, SHADOW_OFFSET[0]], [0, 1, SHADOW_OFFSET[1]]])
        shadow = cv2.warpAffine(shadow, M, (mask.shape[1], mask.shape[0]))
        
        # Apply Gaussian blur for soft shadow
        shadow = cv2.GaussianBlur(shadow, (SHADOW_BLUR_RADIUS*2+1, SHADOW_BLUR_RADIUS*2+1), 0)
        
        # Reduce shadow intensity
        shadow = (shadow * SHADOW_OPACITY).astype(np.uint8)
        
        return shadow
    
    def _apply_texture(self, shape: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply subtle texture to the shape for realism."""
        h, w = shape.shape[:2]
        
        # Generate Perlin-like noise texture
        texture = np.random.randint(200, 256, (h, w), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (5, 5), 0)
        
        # Create subtle fabric-like pattern
        for i in range(0, h, 4):
            texture[i:i+2, :] = np.clip(texture[i:i+2, :] * 0.95, 0, 255)
        
        # Apply texture only where mask is active
        textured_shape = shape.copy()
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Blend texture with shape
        texture_3d = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
        texture_effect = (texture_3d / 255.0 - 0.5) * TEXTURE_OPACITY + 1.0
        textured_shape = (textured_shape * texture_effect * mask_3d).astype(np.uint8)
        
        return textured_shape
    
    def _blend_shadow(self, background: np.ndarray, shadow: np.ndarray, 
                     x: int, y: int) -> np.ndarray:
        """Blend shadow with background."""
        h, w = shadow.shape
        bg_h, bg_w = background.shape[:2]
        
        # Calculate valid region
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bg_w, x + w)
        y2 = min(bg_h, y + h)
        
        if x1 >= x2 or y1 >= y2:
            return background
        
        # Extract regions
        shadow_region = shadow[y1-y:y2-y, x1-x:x2-x]
        bg_region = background[y1:y2, x1:x2]
        
        # Apply shadow (darken background)
        shadow_mask = shadow_region / 255.0
        darkened = bg_region * (1 - shadow_mask[:, :, np.newaxis] * 0.5)
        background[y1:y2, x1:x2] = darkened.astype(np.uint8)
        
        return background
    
    def _blend_shape(self, background: np.ndarray, shape: np.ndarray, mask: np.ndarray,
                    x: int, y: int) -> np.ndarray:
        """Blend shape with background using the mask."""
        h, w = shape.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # Calculate valid region
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bg_w, x + w)
        y2 = min(bg_h, y + h)
        
        if x1 >= x2 or y1 >= y2:
            return background
        
        # Extract regions
        shape_region = shape[y1-y:y2-y, x1-x:x2-x]
        mask_region = mask[y1-y:y2-y, x1-x:x2-x]
        bg_region = background[y1:y2, x1:x2]
        
        # Calculate local brightness adjustment
        bg_brightness = np.mean(bg_region)
        shape_brightness = np.mean(shape_region[mask_region > 0])
        
        if shape_brightness > 0:
            brightness_ratio = bg_brightness / shape_brightness
            brightness_ratio = np.clip(brightness_ratio, 0.7, 1.3)
            shape_region = (shape_region * brightness_ratio).astype(np.uint8)
        
        # Blend using mask with slight transparency
        mask_3d = cv2.cvtColor(mask_region, cv2.COLOR_GRAY2BGR) / 255.0 * SHAPE_OPACITY
        blended = bg_region * (1 - mask_3d) + shape_region * mask_3d
        background[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        return background
    
    def _calculate_bounding_box(self, mask: np.ndarray, x_offset: int, 
                               y_offset: int) -> Tuple[float, float, float, float]:
        """Calculate YOLO format bounding box from mask."""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # Adjust for placement offset
        x += x_offset
        y += y_offset
        
        # Convert to YOLO format (normalized center coordinates)
        x_center = x + w / 2
        y_center = y + h / 2
        
        return (x_center, y_center, w, h)
    
    def _save_results(self, image: np.ndarray, annotations: List[Dict], base_name: str):
        """Save image and YOLO format annotations."""
        # Save image
        image_path = self.images_dir / f"{base_name}.jpg"
        cv2.imwrite(str(image_path), image)
        
        # Save annotations
        label_path = self.labels_dir / f"{base_name}.txt"
        with open(label_path, 'w') as f:
            for ann in annotations:
                class_id = ann['class_id']
                x_center, y_center, w, h = ann['bbox']
                img_w, img_h = ann['image_width'], ann['image_height']
                
                # Normalize coordinates
                x_center_norm = x_center / img_w
                y_center_norm = y_center / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # Write YOLO format line
                f.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} "
                       f"{w_norm:.6f} {h_norm:.6f}\n")


def main():
    """Main function to parse arguments and run the generator."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset for YOLO object detection with realistic shapes on CanSat drone imagery"
    )
    parser.add_argument('--input_dir', type=str, default='aerial_earth_images',
                       help='Path to directory containing aerial earth observation images (default: aerial_earth_images)')
    parser.add_argument('--output_dir', type=str, default='synthetic_cansat_dataset',
                       help='Path to output directory for generated dataset (default: synthetic_cansat_dataset)')
    parser.add_argument('--num_images', type=int, default=500,
                       help='Number of synthetic images to generate (default: 500)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory {args.input_dir} does not exist!")
        return
    
    # Create generator and run
    generator = SyntheticDataGenerator(args.input_dir, args.output_dir, args.num_images)
    generator.generate_dataset()


if __name__ == "__main__":
    main()