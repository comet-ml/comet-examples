"""
Utility functions for the multimodal classification demo.
Handles image processing, JSON parsing, and custom metrics.
"""

import base64
import io
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import requests
from opik.evaluation.metrics import BaseMetric, score_result

logger = logging.getLogger(__name__)


def download_image_from_url(url: str) -> bytes:
    """
    Download an image from a URL and return as bytes.

    Args:
        url: URL to the image

    Returns:
        Image bytes
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.warning(f"Could not download image from {url}: {str(e)}")
        # Fallback: create a simple colored image
        img = Image.new("RGB", (400, 400), color="gray")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return img_bytes.read()


def encode_image_bytes_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string.

    Args:
        image_bytes: Image data as bytes

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from model response, handling various formats.

    Args:
        response: Model response string

    Returns:
        Parsed JSON as dict or None if parsing fails
    """
    if not response:
        return None

    try:
        # Direct JSON parse
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
            # Generic code block
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
    """
    Extract JSON fields to flat column structure for Opik dataset.

    Args:
        json_data: Nested JSON response

    Returns:
        Flat dict with column names as keys
    """
    columns = {}

    # Extract standard fields
    if "label" in json_data:
        columns["response_label"] = json_data["label"]

    if "reason" in json_data:
        columns["response_reason"] = json_data["reason"]

    # Extract optional fields if present
    if "confidence" in json_data:
        columns["response_confidence"] = json_data["confidence"]

    if "categories" in json_data:
        # Join list into string for column storage
        columns["response_categories"] = ", ".join(json_data.get("categories", []))

    if "visual_features" in json_data:
        columns["response_visual_features"] = ", ".join(json_data.get("visual_features", []))

    return columns


class ImageClassificationQualityMetric(BaseMetric):
    """Custom metric for evaluating image classification quality."""

    def __init__(self, name: str = "classification_quality"):
        super().__init__(name=name)

    def score(self, output: Any, **kwargs) -> score_result.ScoreResult:
        """
        Score the image classification based on format and content quality.

        Args:
            output: Model output - can be dict or wrapped in another dict
            **kwargs: Additional context including expected values

        Returns:
            ScoreResult with value 0-1 and explanation
        """
        try:
            # Extract expected values
            expected_label = kwargs.get("expected_label", "").lower()

            # Handle nested output structure
            if isinstance(output, dict) and "output" in output:
                # Output is wrapped - extract the nested dict
                output_data = output["output"]
            else:
                output_data = output

            # Extract actual values from output
            if isinstance(output_data, dict):
                actual_label = (output_data.get("response_label") or output_data.get("label", "")).lower()
                actual_reason = output_data.get("response_reason") or output_data.get("reason", "")
            else:
                # Handle string outputs
                parsed = parse_json_response(str(output_data))
                if parsed:
                    actual_label = parsed.get("label", "").lower()
                    actual_reason = parsed.get("reason", "")
                else:
                    return score_result.ScoreResult(value=0.0, name=self.name, reason="Failed to parse JSON response")

            # Skip scoring if response indicates error
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
                # If no expected label, give partial credit
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


def create_synthetic_image_data() -> List[Tuple[str, Dict[str, str]]]:
    """
    Create a list of image URLs with metadata for synthetic dataset.

    Returns:
        List of tuples (url, metadata_dict)
    """
    image_sources = [
        {
            "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
            "description": "Majestic mountain landscape with snow peaks",
            "category": "nature",
            "expected_label": "highly recommend",
            "expected_reason": "Stunning natural scenery with breathtaking mountain views",
        },
        {
            "url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",
            "description": "Cute cat wearing sunglasses",
            "category": "pet",
            "expected_label": "highly recommend",
            "expected_reason": "Adorable and humorous pet content that appeals to wide audience",
        },
        {
            "url": "https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445",
            "description": "Delicious pizza with fresh toppings",
            "category": "food",
            "expected_label": "recommend",
            "expected_reason": "Appetizing food presentation that looks delicious",
        },
        {
            "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d",
            "description": "Professional headshot of a business person",
            "category": "portrait",
            "expected_label": "neutral",
            "expected_reason": "Standard professional portrait without distinctive features",
        },
        {
            "url": "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b",
            "description": "Laptop with code on screen",
            "category": "technology",
            "expected_label": "neutral",
            "expected_reason": "Generic technology image without special appeal",
        },
        {
            "url": "https://images.unsplash.com/photo-1573152143286-0c422b4d2175",
            "description": "Crowded city street",
            "category": "urban",
            "expected_label": "not recommend",
            "expected_reason": "Busy, chaotic scene that may be overwhelming",
        },
        {
            "url": "https://images.unsplash.com/photo-1472214103451-9374bd1c798e",
            "description": "Peaceful lake at sunset",
            "category": "nature",
            "expected_label": "highly recommend",
            "expected_reason": "Serene and calming natural beauty",
        },
        {
            "url": "https://images.unsplash.com/photo-1518717758536-85ae29035b6d",
            "description": "Happy dog playing in park",
            "category": "pet",
            "expected_label": "highly recommend",
            "expected_reason": "Joyful pet content that brings positive emotions",
        },
        {
            "url": "https://images.unsplash.com/photo-1494790108377-be9c29b29330",
            "description": "Smiling woman portrait",
            "category": "portrait",
            "expected_label": "recommend",
            "expected_reason": "Warm, friendly portrait with positive energy",
        },
        {
            "url": "https://images.unsplash.com/photo-1517849845537-4d257902454a",
            "description": "Close-up of a dog's face",
            "category": "pet",
            "expected_label": "highly recommend",
            "expected_reason": "Endearing pet portrait with expressive features",
        },
    ]

    return [(item["url"], item) for item in image_sources]
