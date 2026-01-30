from PIL import Image
from numpy import ndarray, stack
from io import BytesIO
from base64 import b64encode
from logging import getLogger
from pathlib import Path

from cv2 import imencode, rectangle, putText, FONT_HERSHEY_SIMPLEX, LINE_AA, imwrite
from numpy import ndarray

logger = getLogger(__name__)

# Color mapping for different object labels and teams
LABEL_COLORS = {
    "player": (0, 255, 0),      # Green
    "ball": (0, 0, 255),        # Red  
    "goalkeeper": (255, 255, 0), # Cyan
    "referee": (255, 0, 255),   # Magenta
    "default": (255, 255, 255), # White
}

TEAM_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 165, 255),
    "purple": (128, 0, 128),
    "pink": (203, 192, 255),
    "default": (128, 128, 128),
}

def pil_from_array(array: ndarray) -> Image.Image:
    """
    Converts a frame array (H,W,3 or H,W,4) to PIL Image without resizing.
    """
    if array.ndim == 2:
        # grayscale → convert to RGB for consistency
        array = stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        # drop alpha if present
        array = array[..., :3]
    return Image.fromarray(array)


def image_to_base64(img: Image.Image, fmt: str, quality: int, optimise: bool) -> str:
    buffer = BytesIO()
    img.save(buffer, format=fmt, quality=quality, optimise=optimise)
    return b64encode(buffer.getvalue()).decode("ascii")


def image_to_b64string(image: ndarray) -> str | None:
    try:
        _, image_buffer = imencode(".png", image)
        b64_image = b64encode(image_buffer.tobytes()).decode("utf-8")
        return b64_image
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")


def images_to_b64strings(images: list[ndarray]) -> list[str]:
    b64_images = []
    for image in images:
        b64_image = image_to_b64string(image=image)
        if b64_image:
            b64_images.append(b64_image)
    return b64_images
def draw_annotations_on_frame(
    frame: ndarray,
    annotation,  # FrameAnnotation
    draw_labels: bool = True,
    line_thickness: int = 2,
) -> ndarray:
    """
    Draw bounding boxes and labels on a frame based on FrameAnnotation.
    
    Args:
        frame: The image frame (BGR format from OpenCV)
        annotation: FrameAnnotation object with bboxes and category
        draw_labels: Whether to draw text labels on boxes
        line_thickness: Thickness of bounding box lines
        
    Returns:
        Frame with annotations drawn on it
    """
    annotated = frame.copy()
    
    for bbox in annotation.bboxes:
        x_min, y_min, x_max, y_max = bbox.bbox_2d
        
        # Get color based on label
        label_str = str(bbox.label.value).lower() if hasattr(bbox.label, 'value') else str(bbox.label).lower()
        color = LABEL_COLORS.get(label_str, LABEL_COLORS["default"])
        
        # Try to use team/cluster color if available
        if hasattr(bbox, 'cluster_id') and bbox.cluster_id:
            cluster_str = str(bbox.cluster_id.value).lower() if hasattr(bbox.cluster_id, 'value') else str(bbox.cluster_id).lower()
            color = TEAM_COLORS.get(cluster_str, color)
        
        # Draw rectangle
        rectangle(annotated, (x_min, y_min), (x_max, y_max), color, line_thickness)
        
        # Draw label text
        if draw_labels:
            label_text = label_str
            if hasattr(bbox, 'cluster_id') and bbox.cluster_id:
                cluster_str = str(bbox.cluster_id.value) if hasattr(bbox.cluster_id, 'value') else str(bbox.cluster_id)
                label_text = f"{label_str} ({cluster_str})"
            
            # Draw background for text
            (text_w, text_h), _ = cv2_getTextSize(label_text, FONT_HERSHEY_SIMPLEX, 0.5, 1)
            rectangle(annotated, (x_min, y_min - text_h - 4), (x_min + text_w, y_min), color, -1)
            putText(annotated, label_text, (x_min, y_min - 2), FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, LINE_AA)
    
    # Draw action category at top of frame
    if hasattr(annotation, 'category') and annotation.category:
        category_str = str(annotation.category.value) if hasattr(annotation.category, 'value') else str(annotation.category)
        confidence = getattr(annotation, 'confidence', 0)
        action_text = f"Action: {category_str} ({confidence}%)"
        putText(annotated, action_text, (10, 30), FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, LINE_AA)
    
    return annotated


def cv2_getTextSize(text, font, scale, thickness):
    """Wrapper for cv2.getTextSize"""
    from cv2 import getTextSize
    return getTextSize(text, font, scale, thickness)


def save_annotated_frames(
    pseudo_gt_annotations: list,  # list[PseudoGroundTruth]
    output_dir: str | Path,
    prefix: str = "annotated",
) -> list[Path]:
    """
    Save frames with bounding boxes drawn from PseudoGroundTruth annotations.
    
    Args:
        pseudo_gt_annotations: List of PseudoGroundTruth objects
        output_dir: Directory to save annotated images
        prefix: Filename prefix
        
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for pgt in pseudo_gt_annotations:
        frame_num = pgt.frame_number
        frame = pgt.spatial_image
        annotation = pgt.annotation
        
        # Draw annotations on frame
        annotated_frame = draw_annotations_on_frame(frame, annotation)
        
        # Save annotated frame
        output_path = output_dir / f"{prefix}_{frame_num:06d}.png"
        success = imwrite(str(output_path), annotated_frame)
        
        if success:
            saved_paths.append(output_path)
            logger.info(f"Saved annotated frame {frame_num} to {output_path}")
        else:
            logger.error(f"Failed to save annotated frame {frame_num}")
        
        # Also save original frame for comparison
        original_path = output_dir / f"{prefix}_{frame_num:06d}_original.png"
        imwrite(str(original_path), frame)
    
    return saved_paths
