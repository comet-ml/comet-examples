"""Utility functions for the multimodal classification demo."""

import base64
import io
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import requests
from opik.evaluation.metrics import BaseMetric, score_result


def download_image_from_url(url: str, max_size: tuple = (256, 256)) -> Image:
    """Download an image from a URL and return as PIL Image."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")
        
        return img
    except Exception:
        return Image.new("RGB", (400, 400), color="gray")


def encode_base64_uri_from_pil(image: Image) -> str:
    """Convert a PIL image to a base64 data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from model response, handling various formats."""
    if not response:
        return None

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                json_str = response[start:end].strip()
                return json.loads(json_str)
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                json_str = response[start:end].strip()
                try:
                    return json.loads(json_str)
                except:
                    pass

        # Try to find JSON-like structure
        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx : end_idx + 1]
                return json.loads(json_str)
        except:
            pass

    return None


def extract_json_fields_to_columns(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract JSON fields to flat column structure for Opik dataset."""
    columns = {}
    
    if "label" in json_data:
        columns["response_label"] = json_data["label"]
    if "reason" in json_data:
        columns["response_reason"] = json_data["reason"]
    if "confidence" in json_data:
        columns["response_confidence"] = json_data["confidence"]
    if "categories" in json_data:
        columns["response_categories"] = ", ".join(json_data.get("categories", []))
    if "visual_features" in json_data:
        columns["response_visual_features"] = ", ".join(json_data.get("visual_features", []))
    
    return columns


class ImageClassificationQualityMetric(BaseMetric):
    """Custom metric for evaluating image classification quality."""

    def __init__(self, name: str = "classification_quality"):
        super().__init__(name=name)

    def score(self, output: Any, **kwargs) -> score_result.ScoreResult:
        """Score the image classification based on format and content quality."""
        try:
            expected_label = kwargs.get("expected_label", "").lower()
            
            if isinstance(output, dict) and "output" in output:
                output_data = output["output"]
            else:
                output_data = output

            if isinstance(output_data, dict):
                actual_label = (output_data.get("response_label") or output_data.get("label", "")).lower()
                actual_reason = output_data.get("response_reason") or output_data.get("reason", "")
            else:
                parsed = parse_json_response(str(output_data))
                if parsed:
                    actual_label = parsed.get("label", "").lower()
                    actual_reason = parsed.get("reason", "")
                else:
                    return score_result.ScoreResult(value=0.0, name=self.name, reason="Failed to parse JSON response")

            if not actual_label or not actual_reason:
                return score_result.ScoreResult(value=0.0, name=self.name, reason="Missing label or reason in response")

            score = 0.0
            reasons = []

            # Check label validity (40% of score)
            valid_labels = ["highly recommend", "recommend", "neutral", "not recommend", "strongly not recommend"]
            if actual_label in valid_labels:
                score += 0.4
                reasons.append(f"✓ Valid label: {actual_label}")
            else:
                reasons.append(f"✗ Invalid label: {actual_label}")

            # Check label accuracy if expected label provided (30% of score)
            if expected_label and actual_label == expected_label:
                score += 0.3
                reasons.append("✓ Label matches expected")
            elif expected_label:
                reasons.append(f"✗ Expected '{expected_label}', got '{actual_label}'")
            else:
                score += 0.15

            # Check reason quality (30% of score)
            if actual_reason and len(actual_reason.strip()) > 20:
                score += 0.3
                reasons.append("✓ Detailed reason provided")
            elif actual_reason:
                score += 0.15
                reasons.append("○ Brief reason provided")
            else:
                reasons.append("✗ No reason provided")

            return score_result.ScoreResult(value=score, name=self.name, reason=" | ".join(reasons))

        except Exception as e:
            return score_result.ScoreResult(value=0.0, name=self.name, reason=f"Scoring error: {str(e)}")


def create_synthetic_image_data() -> pd.DataFrame:
    """Create a tabular dataset with image URLs and metadata."""
    image_data = pd.DataFrame([
        {
            "image_id": "IMG_001",
            "image_url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
            "description": "Majestic mountain landscape with snow peaks",
            "category": "nature",
            "expected_label": "highly recommend",
            "expected_reason": "Stunning natural scenery with breathtaking mountain views",
            "upload_date": "2024-01-15",
            "file_size_mb": 2.3,
            "resolution": "1920x1080",
            "source": "MANUAL"
        },
        {
            "image_id": "IMG_002", 
            "image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",
            "description": "Cute cat wearing sunglasses",
            "category": "pet",
            "expected_label": "highly recommend",
            "expected_reason": "Adorable and humorous pet content that appeals to wide audience",
            "upload_date": "2024-01-16",
            "file_size_mb": 1.8,
            "resolution": "1600x1200",
            "source": "MANUAL"
        },
        {
            "image_id": "IMG_003",
            "image_url": "https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445",
            "description": "Delicious pizza with fresh toppings",
            "category": "food",
            "expected_label": "recommend",
            "expected_reason": "Appetizing food presentation that looks delicious",
            "upload_date": "2024-01-17",
            "file_size_mb": 2.1,
            "resolution": "2048x1536",
            "source": "MANUAL"
        },
        {
            "image_id": "IMG_004",
            "image_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d",
            "description": "Professional headshot of a business person",
            "category": "portrait",
            "expected_label": "neutral",
            "expected_reason": "Standard professional portrait without distinctive features",
            "upload_date": "2024-01-18",
            "file_size_mb": 1.5,
            "resolution": "1200x1600",
            "source": "MANUAL"
        },
        {
            "image_id": "IMG_005",
            "image_url": "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b",
            "description": "Laptop with code on screen",
            "category": "technology",
            "expected_label": "neutral",
            "expected_reason": "Generic technology image without special appeal",
            "upload_date": "2024-01-19",
            "file_size_mb": 1.9,
            "resolution": "1920x1080",
            "source": "MANUAL"
        },
        {
            "image_id": "IMG_006",
            "image_url": "https://images.unsplash.com/photo-1573152143286-0c422b4d2175",
            "description": "Crowded city street",
            "category": "urban",
            "expected_label": "not recommend",
            "expected_reason": "Busy, chaotic scene that may be overwhelming",
            "upload_date": "2024-01-20",
            "file_size_mb": 2.4,
            "resolution": "2560x1440",
            "source": "MANUAL"
        },
        {
            "image_id": "IMG_007",
            "image_url": "https://images.unsplash.com/photo-1472214103451-9374bd1c798e",
            "description": "Peaceful lake at sunset",
            "category": "nature",
            "expected_label": "highly recommend",
            "expected_reason": "Serene and calming natural beauty",
            "upload_date": "2024-01-21",
            "file_size_mb": 2.7,
            "resolution": "3000x2000",
            "source": "MANUAL"
        },
        {
            "image_id": "IMG_008",
            "image_url": "https://images.unsplash.com/photo-1518717758536-85ae29035b6d",
            "description": "Happy dog playing in park",
            "category": "pet",
            "expected_label": "highly recommend",
            "expected_reason": "Joyful pet content that brings positive emotions",
            "upload_date": "2024-01-22",
            "file_size_mb": 2.0,
            "resolution": "1800x1350",
            "source": "MANUAL"
        },
        {
            "image_id": "IMG_009",
            "image_url": "https://images.unsplash.com/photo-1494790108377-be9c29b29330",
            "description": "Smiling woman portrait",
            "category": "portrait",
            "expected_label": "recommend",
            "expected_reason": "Warm, friendly portrait with positive energy",
            "upload_date": "2024-01-23",
            "file_size_mb": 1.7,
            "resolution": "1500x2000",
            "source": "MANUAL"
        },
        {
            "image_id": "IMG_010",
            "image_url": "https://images.unsplash.com/photo-1517849845537-4d257902454a",
            "description": "Close-up of a dog's face",
            "category": "pet",
            "expected_label": "highly recommend",
            "expected_reason": "Endearing pet portrait with expressive features",
            "upload_date": "2024-01-24",
            "file_size_mb": 1.6,
            "resolution": "1400x1400",
            "source": "MANUAL"
        }
    ])
    
    return image_data


def process_image_dataset(base_df: pd.DataFrame, num_items: int = 100) -> pd.DataFrame:
    """Process the base DataFrame and create expanded dataset with image processing."""
    processed_images = []
    
    # Process each image from the base DataFrame
    for idx, row in base_df.iterrows():
        try:
            # COMMENTED OUT: Base64 encoding for faster testing
            # img = download_image_from_url(row['image_url'])
            # image_base64 = encode_base64_uri_from_pil(img)
            
            processed_images.append({
                "image_id": row['image_id'],
                "image_url": row['image_url'],  # Using URL directly
                "description": row['description'],
                "category": row['category'],
                "expected_label": row['expected_label'],
                "expected_reason": row['expected_reason'],
                "upload_date": row['upload_date'],
                "file_size_mb": row['file_size_mb'],
                "resolution": row['resolution'],
                "source": row['source']
            })
        except Exception:
            continue
    
    # Create expanded dataset by cycling through processed images
    rows = []
    for i in range(num_items):
        base_idx = i % len(processed_images)
        base_item = processed_images[base_idx]
        variation = i // len(processed_images) + 1
        
        row = {
            "item_id": f"img_{i:04d}",
            "variation": variation,
            "image_id": base_item["image_id"],
            # "image_base64": base_item["image_base64"],
            "image_url": base_item["image_url"],
            "description": base_item["description"],
            "content_description": f"{base_item['description']} (v{variation})",
            "category": base_item["category"],
            "expected_label": base_item["expected_label"],
            "expected_reason": base_item["expected_reason"],
            "upload_date": base_item["upload_date"],
            "file_size_mb": base_item["file_size_mb"],
            "resolution": base_item["resolution"],
            "source": base_item["source"]
        }
        rows.append(row)
    
    return pd.DataFrame(rows)
