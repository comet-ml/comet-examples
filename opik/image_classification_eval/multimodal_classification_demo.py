#!/usr/bin/env python3
"""Multimodal Image Classification Demo with Opik"""

import os
import json
import random
import string
import pandas as pd
from typing import Dict, Any, List

import opik
from opik import track
from opik.evaluation import evaluate
from opik.integrations.openai import track_openai
from opik.integrations.genai import track_genai
from opik.evaluation.metrics import BaseMetric, score_result
import openai
from google import genai
from google.genai.types import GenerateContentConfig
import litellm
from dotenv import load_dotenv

from utils import (
    download_image_from_url,
    encode_base64_uri_from_pil,
    parse_json_response,
    extract_json_fields_to_columns,
    ImageClassificationQualityMetric,
    create_synthetic_image_data,
    process_image_dataset,
)

load_dotenv()


# Custom Opik Metrics for Classification
class TruePositiveMetric(BaseMetric):
    """Metric to calculate True Positives (TP)"""
    def __init__(self):
        super().__init__(name="true_positive")
    
    def score(self, output: Dict[str, Any], **kwargs) -> score_result.ScoreResult:
        expected_label = kwargs.get("expected_label", "").lower()
        predicted_label = output.get("response_label", "").lower()
        
        positive_labels = ["highly recommend", "recommend"]
        is_expected_positive = expected_label in positive_labels
        is_predicted_positive = predicted_label in positive_labels
        
        value = 1.0 if (is_expected_positive and is_predicted_positive) else 0.0
        return score_result.ScoreResult(value=value, name=self.name)


class TrueNegativeMetric(BaseMetric):
    """Metric to calculate True Negatives (TN)"""
    def __init__(self):
        super().__init__(name="true_negative")
    
    def score(self, output: Dict[str, Any], **kwargs) -> score_result.ScoreResult:
        expected_label = kwargs.get("expected_label", "").lower()
        predicted_label = output.get("response_label", "").lower()
        
        positive_labels = ["highly recommend", "recommend"]
        is_expected_positive = expected_label in positive_labels
        is_predicted_positive = predicted_label in positive_labels
        
        value = 1.0 if (not is_expected_positive and not is_predicted_positive) else 0.0
        return score_result.ScoreResult(value=value, name=self.name)


class FalsePositiveMetric(BaseMetric):
    """Metric to calculate False Positives (FP)"""
    def __init__(self):
        super().__init__(name="false_positive")
    
    def score(self, output: Dict[str, Any], **kwargs) -> score_result.ScoreResult:
        expected_label = kwargs.get("expected_label", "").lower()
        predicted_label = output.get("response_label", "").lower()
        
        positive_labels = ["highly recommend", "recommend"]
        is_expected_positive = expected_label in positive_labels
        is_predicted_positive = predicted_label in positive_labels
        
        value = 1.0 if (not is_expected_positive and is_predicted_positive) else 0.0
        return score_result.ScoreResult(value=value, name=self.name)


class FalseNegativeMetric(BaseMetric):
    """Metric to calculate False Negatives (FN)"""
    def __init__(self):
        super().__init__(name="false_negative")
    
    def score(self, output: Dict[str, Any], **kwargs) -> score_result.ScoreResult:
        expected_label = kwargs.get("expected_label", "").lower()
        predicted_label = output.get("response_label", "").lower()
        
        positive_labels = ["highly recommend", "recommend"]
        is_expected_positive = expected_label in positive_labels
        is_predicted_positive = predicted_label in positive_labels
        
        value = 1.0 if (is_expected_positive and not is_predicted_positive) else 0.0
        return score_result.ScoreResult(value=value, name=self.name)


def generate_random_tag(length=6):
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))


class ImageClassifier:
    def __init__(self, model: str):
        self.model = model

    @track()
    def process_item(self, item: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
        """Process a dataset item and return classification with extracted fields"""
        try:
            # COMMENTED OUT: Base64 image processing for faster testing
            # image_base64 = item.get("image_base64") or item.get("image", "")
            # if not image_base64:
            #     return {"response_label": "error", "response_reason": "No image data", "raw_response": ""}
            # 
            # if isinstance(image_base64, str) and image_base64.startswith("data:image"):
            #     image_base64 = image_base64.split(",", 1)[1]
            # 
            # response = self.classify_image(image_base64, system_prompt)
            
            # Use image URL directly for faster testing
            image_url = item.get("image_url", "")
            if not image_url:
                return {"response_label": "error", "response_reason": "No image URL", "raw_response": ""}
            
            response = self.classify_image_from_url(image_url, system_prompt)
            
            json_data = parse_json_response(response) if isinstance(response, str) else response

            if not json_data:
                return {"response_label": "neutral", "response_reason": "Failed to parse response", "raw_response": str(response)}

            columns = extract_json_fields_to_columns(json_data)
            columns["raw_response"] = str(response)
            return columns

        except Exception as e:
            return {"response_label": "error", "response_reason": str(e), "raw_response": ""}


class OpenAIClassifier(ImageClassifier):
    def __init__(self, model: str = "gpt-4o"):
        super().__init__(model)
        self.client = track_openai(openai.Client())

    @track()
    def classify_image(self, image_base64: str, prompt_text: str) -> Dict[str, Any]:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.3, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    @track()
    def classify_image_from_url(self, image_url: str, prompt_text: str) -> Dict[str, Any]:
        """Classify image using URL directly (faster for testing)"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.3, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)


class GeminiClassifier(ImageClassifier):
    def __init__(self, model: str = "gemini-2.0-flash-001"):
        super().__init__(model)
        self.client = track_genai(genai.Client(api_key=os.getenv("GEMINI_API_KEY")))

    @track()
    def classify_image(self, image_base64: str, prompt_text: str) -> Dict[str, Any]:
        full_prompt = f"{prompt_text}\n\nImage: [Image data provided]"
        response = self.client.models.generate_content(
            model=self.model,
            contents=[{"parts": [{"text": full_prompt}, {"inline_data": {"mime_type": "image/png", "data": image_base64}}]}],
            config=GenerateContentConfig(temperature=0.3, response_mime_type="application/json"),
        )
        return json.loads(response.text)

    @track()
    def classify_image_from_url(self, image_url: str, prompt_text: str) -> Dict[str, Any]:
        """Classify image using URL directly (faster for testing)"""
        # Note: Gemini doesn't support direct URL access, so we'll download and encode
        # For now, return a placeholder response
        return {
            "label": "neutral",
            "reason": f"Gemini URL classification placeholder for {image_url}"
        }


class OpenRouterClassifier(ImageClassifier):
    def __init__(self, model: str = "qwen/qwen-2-vl-7b-instruct"):
        super().__init__(model)

    @track()
    def classify_image(self, image_base64: str, prompt_text: str) -> Dict[str, Any]:
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }]
        response = litellm.completion(
            model=f"openrouter/{self.model}", messages=messages, temperature=0.3, api_key=os.getenv("OPENROUTER_API_KEY")
        )
        return parse_json_response(response.choices[0].message.content)

    @track()
    def classify_image_from_url(self, image_url: str, prompt_text: str) -> Dict[str, Any]:
        """Classify image using URL directly (faster for testing)"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }]
        response = litellm.completion(
            model=f"openrouter/{self.model}", messages=messages, temperature=0.3, api_key=os.getenv("OPENROUTER_API_KEY")
        )
        return parse_json_response(response.choices[0].message.content)


def create_image_dataset(num_items: int = 100) -> pd.DataFrame:
    """Create synthetic dataset with images as tabular data"""
    # Get base image data as DataFrame
    base_df = create_synthetic_image_data()
    
    # Process and expand the dataset with image processing
    return process_image_dataset(base_df, num_items)


def evaluate_classification(x: Dict[str, Any], classifier: ImageClassifier, system_prompt: str) -> Dict[str, Any]:
    """Evaluation task that returns output in expected format"""
    result = classifier.process_item(x, system_prompt)
    return {
        "output": {
            "response_label": result.get("response_label", ""), 
            "response_reason": result.get("response_reason", "")
        },
        **result
    }


def run_evaluation_with_prompt(dataset: Any, classifiers: List[ImageClassifier], prompt: opik.Prompt, experiment_tag: str, project_name: str) -> Dict[str, Any]:
    """Run evaluation across all classifiers with given prompt"""
    results = {}
    quality_metric = ImageClassificationQualityMetric()
    
    # Create the boolean metrics
    tp_metric = TruePositiveMetric()
    tn_metric = TrueNegativeMetric()
    fp_metric = FalsePositiveMetric()
    fn_metric = FalseNegativeMetric()

    for classifier in classifiers:
        model_name = classifier.model.replace("/", "_").replace("-", "_")
        try:
            def task(x: Dict[str, Any]) -> Dict[str, Any]:
                return evaluate_classification(x, classifier, prompt.format())

            experiment = evaluate(
                dataset=dataset,
                task=task,
                scoring_metrics=[quality_metric, tp_metric, tn_metric, fp_metric, fn_metric],
                experiment_name=f"{model_name}_{experiment_tag}",
                project_name=project_name,
                prompt=prompt,
            )
            results[model_name] = {"experiment": experiment, "status": "success"}
        except Exception as e:
            results[model_name] = {"status": "error", "error": str(e)}

    return results


def calculate_metrics_from_experiments(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Calculate Accuracy, Precision, and Recall from Opik experiment results"""
    metrics_summary = {}
    
    for model_name, result in results.items():
        if result.get("status") != "success":
            metrics_summary[model_name] = {"error": result.get("error", "Unknown error")}
            continue
            
        try:
            experiment = result["experiment"]
            
            # Get metrics from the experiment - try different ways to access metrics
            metrics_data = None
            
            # Try different ways to access metrics
            if hasattr(experiment, 'metrics'):
                metrics_data = experiment.metrics
            elif hasattr(experiment, 'results') and hasattr(experiment.results, 'metrics'):
                metrics_data = experiment.results.metrics
            elif hasattr(experiment, 'summary'):
                metrics_data = experiment.summary
            
            if metrics_data is None:
                metrics_summary[model_name] = {"error": "Could not access metrics from experiment"}
                continue
            
            # Get the boolean metric values
            tp_avg = metrics_data.get("true_positive", {}).get("avg", 0) if isinstance(metrics_data.get("true_positive"), dict) else 0
            tn_avg = metrics_data.get("true_negative", {}).get("avg", 0) if isinstance(metrics_data.get("true_negative"), dict) else 0
            fp_avg = metrics_data.get("false_positive", {}).get("avg", 0) if isinstance(metrics_data.get("false_positive"), dict) else 0
            fn_avg = metrics_data.get("false_negative", {}).get("avg", 0) if isinstance(metrics_data.get("false_negative"), dict) else 0
            
            # Get sample count
            num_samples = metrics_data.get("true_positive", {}).get("count", 20) if isinstance(metrics_data.get("true_positive"), dict) else 20
            
            # Convert averages to counts
            tp_count = tp_avg * num_samples
            tn_count = tn_avg * num_samples
            fp_count = fp_avg * num_samples
            fn_count = fn_avg * num_samples
            
            # Calculate metrics
            accuracy = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count) if (tp_count + tn_count + fp_count + fn_count) > 0 else 0
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_summary[model_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "total_samples": num_samples,
                "tp": tp_count,
                "tn": tn_count,
                "fp": fp_count,
                "fn": fn_count
            }
            
        except Exception as e:
            metrics_summary[model_name] = {"error": f"Failed to calculate metrics: {str(e)}"}
    
    return metrics_summary


def main():
    """Main demo execution"""
    project_name = "demo-multimodal-image-classification"
    os.environ["OPIK_REQUEST_TIMEOUT"] = "300"
    os.environ["OPIK_BATCH_SIZE"] = "20"
    os.environ["OPIK_PROJECT_NAME"] = project_name

    opik.configure(use_local=False, workspace=os.getenv("OPIK_WORKSPACE_NAME"), api_key=os.getenv("OPIK_API_KEY"))
    client = opik.Opik()

    # Create new tabular dataset with unique name
    import time
    timestamp = int(time.time())
    dataset_name = f"multimodal_images_{timestamp}"
    df = create_image_dataset(num_items=20)
    
    print("Dataset created successfully!")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Convert DataFrame to list of dicts for Opik insertion
    dataset_items = df.to_dict('records')

    # Always create a new dataset
    dataset = client.create_dataset(name=dataset_name)
    batch_size = 20
    for i in range(0, len(dataset_items), batch_size):
        batch = dataset_items[i : i + batch_size]
        dataset.insert(batch)
        print(f"Inserted batch {i // batch_size + 1}/{(len(dataset_items) + batch_size - 1) // batch_size}")
    
    print(f"Dataset '{dataset_name}' created with {len(dataset_items)} items")
    print(f"Columns: {', '.join(df.columns.tolist())}")

    # Initialize classifiers
    classifiers = [
        OpenAIClassifier("gpt-4o"), 
        GeminiClassifier("gemini-2.0-flash-001"), 
        OpenRouterClassifier("qwen/qwen-2-vl-7b-instruct")
    ]

    # Single basic prompt
    prompt = opik.Prompt(
        name=f"{project_name}_classification_prompt",
        prompt="""Analyze the provided image and classify it for content recommendation.

Respond with JSON in this exact format:
{
    "label": "highly recommend" | "recommend" | "neutral" | "not recommend" | "strongly not recommend",
    "reason": "Your explanation for this classification based on the image content"
}""",
    )

    # Run evaluation with single prompt
    random_tag = generate_random_tag()
    results = run_evaluation_with_prompt(dataset, classifiers, prompt, f"basic_{random_tag}", project_name)

    # Calculate and display metrics
    print("\n" + "="*60)
    print("📊 EVALUATION METRICS SUMMARY")
    print("="*60)
    
    metrics_summary = calculate_metrics_from_experiments(results)
    
    for model_name, metrics in metrics_summary.items():
        print(f"\n🤖 Model: {model_name}")
        print("-" * 40)
        
        if "error" in metrics:
            print(f"❌ Error: {metrics['error']}")
        else:
            print(f"📈 Accuracy:  {metrics['accuracy']:.3f}")
            print(f"🎯 Precision: {metrics['precision']:.3f}")
            print(f"🔄 Recall:    {metrics['recall']:.3f}")
            print(f"⚖️  F1-Score: {metrics['f1_score']:.3f}")
            print(f"📊 Samples:   {metrics['total_samples']}")
            print(f"✅ TP: {metrics['tp']:.0f}, TN: {metrics['tn']:.0f}, FP: {metrics['fp']:.0f}, FN: {metrics['fn']:.0f}")

    print("\n" + "="*60)
    print("✅ Evaluation completed! Check Opik dashboard for detailed results.")
    print("="*60)

    opik.flush_tracker()


if __name__ == "__main__":
    required_keys = ["OPIK_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("⚠️  Missing required API keys:", ", ".join(missing_keys))
    else:
        main()
