import albumentations as A

def get_military_augmentations():
    """
    Returns an Albumentations composition tailored for battlefield conditions.
    Includes fog, low light, noise, and motion blur.
    """
    return A.Compose([
        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.1, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.1), contrast_limit=(-0.2, 0.2), p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.MotionBlur(blur_limit=7, p=0.3),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.2), # simulates night vision noise slightly
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.2), # Random occlusion
        A.HorizontalFlip(p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
