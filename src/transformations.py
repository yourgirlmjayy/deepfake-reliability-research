from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance

def jpeg_compress(image: Image.Image, quality: int = 50) -> Image.Image:
    """Re-encode an image as JPEG at the requested quality."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def center_crop(image: Image.Image, crop_fraction: float = 0.80) -> Image.Image:
    """Keep the center portion of an image."""
    if not 0 < crop_fraction <= 1:
        raise ValueError("crop_fraction must be in (0, 1].")
    w, h = image.size
    nw, nh = int(w * crop_fraction), int(h * crop_fraction)
    left = (w - nw) // 2
    top = (h - nh) // 2
    return image.crop((left, top, left + nw, top + nh))

def resize_roundtrip(image: Image.Image, scale: float = 0.50) -> Image.Image:
    """Downscale and then resize back to the original dimensions."""
    if not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1].")
    w, h = image.size
    small = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return small.resize((w, h), Image.Resampling.LANCZOS)

def gaussian_blur(image: Image.Image, radius: float = 2.0) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def contrast_filter(image: Image.Image, factor: float = 1.35) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)

def screenshot_like(image: Image.Image, scale: float = 0.72, jpeg_quality: int = 88) -> Image.Image:
    """
    Approximate a common screenshot/re-share pipeline by resampling the image
    and re-encoding it. This is a controlled proxy, not a literal OS screenshot.
    """
    transformed = resize_roundtrip(image, scale=scale)
    return jpeg_compress(transformed, quality=jpeg_quality)
