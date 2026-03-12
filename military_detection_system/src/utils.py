# Place any common utilities here, e.g., drawing bounding boxes manually, calculating GSD (Ground Sample Distance)
def calculate_distance(bbox_height_pixels, actual_height_meters, focal_length_mm, sensor_height_mm, image_height_pixels):
    """
    Rough distance estimation based on object height in pixels.
    """
    distance = (focal_length_mm * actual_height_meters * image_height_pixels) / (bbox_height_pixels * sensor_height_mm)
    return distance
