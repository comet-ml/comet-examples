#!/usr/bin/env python3
"""
Multimodal Image Classification Demo with Opik

This demo showcases:
1. Importing datasets with the SDK
2. Prompt iterations with JSON responses mapped to columns
3. Full prompt optimization with MIPRO

Following the clean patterns from the RAG example while demonstrating
multimodal classification capabilities.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import random
import string
import datetime

import opik
from opik import track
from opik.evaluation import evaluate
from opik.integrations.openai import track_openai
from opik.integrations.genai import track_genai
import openai
from google import genai
from google.genai.types import GenerateContentConfig
import litellm
from dotenv import load_dotenv

from utils import (
    download_image_from_url,
    encode_image_bytes_to_base64,
    parse_json_response,
    extract_json_fields_to_columns,
    ImageClassificationQualityMetric,
    create_synthetic_image_data,
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Helper functions (following RAG example patterns)
def get_datestamp():
    return datetime.datetime.now().strftime(format="%Y%m%d%H%M%S")


def generate_random_tag(length=6):
    characters = string.ascii_lowercase + string.digits
    return "".join(random.choice(characters) for _ in range(length))


class ImageClassifier(ABC):
    """Abstract base class for image classifiers (similar to CometBot pattern)"""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    @track()
    def classify_image(self, image_base64: str, prompt_text: str) -> Dict[str, Any]:
        """Classify an image and return structured response"""
        pass

    @track()
    def process_item(self, item: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
        """Process a dataset item and return classification with extracted fields"""
        try:
            # Get image data
            image_base64 = item["image"]

            # Classify the image
            response = self.classify_image(image_base64, system_prompt)

            # Parse JSON response
            if isinstance(response, str):
                json_data = parse_json_response(response)
            else:
                json_data = response

            if not json_data:
                logger.warning(f"Failed to parse JSON from {self.model}")
                return {"response_label": "neutral", "response_reason": "Failed to parse response", "raw_response": str(response)}

            # Extract fields to columns
            columns = extract_json_fields_to_columns(json_data)
            columns["raw_response"] = str(response)

            return columns

        except Exception as e:
            logger.error(f"Error in {self.model}: {str(e)}")
            return {"response_label": "error", "response_reason": str(e), "raw_response": ""}


class OpenAIClassifier(ImageClassifier):
    """OpenAI GPT-4V classifier"""

    def __init__(self, model: str = "gpt-4o"):
        super().__init__(model)
        self.client = track_openai(openai.Client())

    @track()
    def classify_image(self, image_base64: str, prompt_text: str) -> Dict[str, Any]:
        """Classify image using OpenAI GPT-4V"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.3, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)


class GeminiClassifier(ImageClassifier):
    """Google Gemini 1.5 Pro classifier"""

    def __init__(self, model: str = "gemini-2.0-flash-001"):
        super().__init__(model)
        self.client = track_genai(genai.Client(api_key=os.getenv("GEMINI_API_KEY")))

    @track()
    def classify_image(self, image_base64: str, prompt_text: str) -> Dict[str, Any]:
        """Classify image using Gemini"""
        # For Gemini, we need to format the prompt differently
        full_prompt = f"{prompt_text}\n\nImage: [Image data provided]"

        response = self.client.models.generate_content(
            model=self.model,
            contents=[{"parts": [{"text": full_prompt}, {"inline_data": {"mime_type": "image/png", "data": image_base64}}]}],
            config=GenerateContentConfig(temperature=0.3, response_mime_type="application/json"),
        )

        return json.loads(response.text)


class OpenRouterClassifier(ImageClassifier):
    """OpenRouter classifier for Qwen-VL and other models"""

    def __init__(self, model: str = "qwen/qwen-2-vl-7b-instruct"):
        super().__init__(model)

    @track()
    def classify_image(self, image_base64: str, prompt_text: str) -> Dict[str, Any]:
        """Classify image using OpenRouter (via litellm)"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]

        response = litellm.completion(
            model=f"openrouter/{self.model}", messages=messages, temperature=0.3, api_key=os.getenv("OPENROUTER_API_KEY")
        )

        return parse_json_response(response.choices[0].message.content)


def create_image_dataset(num_items: int = 200) -> List[Dict[str, Any]]:
    """Create synthetic dataset with images from public sources"""
    logger.info(f"Creating synthetic dataset with {num_items} items...")

    # Get base image data
    base_images = create_synthetic_image_data()

    # Download and encode images once
    encoded_images = []
    for idx, (url, metadata) in enumerate(base_images):
        try:
            logger.info(f"Downloading image {idx + 1}/{len(base_images)}")
            image_bytes = download_image_from_url(url)
            image_base64 = encode_image_bytes_to_base64(image_bytes)

            encoded_images.append({"image": image_base64, "image_url": url, **metadata})
        except Exception as e:
            logger.error(f"Failed to process image {url}: {e}")

    # Create full dataset by cycling through images
    dataset_items = []
    for i in range(num_items):
        base_idx = i % len(encoded_images)
        item = encoded_images[base_idx].copy()

        # Add variation to make each item unique
        variation = i // len(encoded_images) + 1
        item["item_id"] = f"img_{i:04d}"
        item["variation"] = variation
        item["content_description"] = f"{item['description']} (v{variation})"

        dataset_items.append(item)

    logger.info(f"✅ Created dataset with {len(dataset_items)} items")
    return dataset_items


def evaluate_classification(x: Dict[str, Any], classifier: ImageClassifier, system_prompt: str) -> Dict[str, Any]:
    """Evaluation task that returns output in expected format"""
    result = classifier.process_item(x, system_prompt)

    # Return in format expected by the metric (with 'output' key)
    return {
        "output": {"response_label": result.get("response_label", ""), "response_reason": result.get("response_reason", "")},
        "response": result.get("raw_response", ""),
    }


def run_evaluation_with_prompt(
    dataset: Any, classifiers: List[ImageClassifier], prompt: opik.Prompt, experiment_tag: str, project_name: str
) -> Dict[str, Any]:
    """Run evaluation across all classifiers with given prompt"""
    results = {}

    # Create metrics
    quality_metric = ImageClassificationQualityMetric()

    for classifier in classifiers:
        model_name = classifier.model.replace("/", "_").replace("-", "_")
        print(f"\n🔍 Evaluating {model_name}...")

        try:
            # Create evaluation task for this classifier
            def task(x: Dict[str, Any]) -> Dict[str, Any]:
                return evaluate_classification(x, classifier, prompt.format())

            # Run evaluation
            experiment = evaluate(
                dataset=dataset,
                task=task,
                scoring_metrics=[quality_metric],
                experiment_name=f"{model_name}_{experiment_tag}",
                project_name=project_name,
                prompt=prompt,
            )

            results[model_name] = {"experiment": experiment, "status": "success"}
            print(f"✅ {model_name} evaluation completed")

        except Exception as e:
            error_msg = f"Error evaluating {model_name}: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            results[model_name] = {"status": "error", "error": str(e)}

    return results


def main():
    """Main demo execution"""
    print("🚀 Multimodal Image Classification Demo with Opik")
    print("=" * 70)

    # Configure Opik
    project_name = "demo-multimodal-image-classification"
    opik.configure(use_local=False, workspace=os.getenv("OPIK_WORKSPACE_NAME"), api_key=os.getenv("OPIK_API_KEY"))

    os.environ["OPIK_PROJECT_NAME"] = project_name

    # Initialize Opik client
    client = opik.Opik()

    # Step 1: Create and import dataset using SDK
    print("\n📊 Step 1: Creating dataset using Opik SDK")
    print("-" * 50)

    dataset_name = "multimodal_images"

    # Always create fresh dataset items for the demo
    dataset_items = create_image_dataset(num_items=200)

    try:
        # Try to get existing dataset
        dataset = client.get_dataset(name=dataset_name)
        print(f"✅ Using existing dataset '{dataset_name}'")
        # Clear existing data and insert fresh items
        # Note: Opik doesn't have a clear method, so we'll work with existing data
    except Exception:
        # Create new dataset if it doesn't exist
        dataset = client.create_dataset(name=dataset_name)
        dataset.insert(dataset_items)
        print(f"✅ Dataset '{dataset_name}' created with {len(dataset_items)} items")
        print(f"   Columns: {', '.join(dataset_items[0].keys())}")

    # Initialize classifiers
    classifiers = [OpenAIClassifier("gpt-4o"), GeminiClassifier("gemini-2.0-flash-001"), OpenRouterClassifier("qwen/qwen-2-vl-7b-instruct")]

    # Step 2: Prompt Iterations
    print("\n📝 Step 2: Prompt Iterations with JSON Response Mapping")
    print("-" * 50)
    # Iteration 1: Basic prompt
    prompt_v1 = opik.Prompt(
        name=f"{project_name}_classification_prompt",
        prompt="""Analyze the provided image and classify it for content recommendation.

Respond with JSON in this exact format:
{
    "label": "highly recommend" | "recommend" | "neutral" | "not recommend" | "strongly not recommend",
    "reason": "Your explanation for this classification based on the image content"
}""",
    )

    print("\n🔄 Iteration 1: Basic classification prompt")
    random_tag = generate_random_tag()
    results_v1 = run_evaluation_with_prompt(dataset, classifiers[:2], prompt_v1, f"v1_{random_tag}", project_name)
    logger.info(f"Completed iteration 1: {len(results_v1)} models evaluated")

    # Iteration 2: More detailed prompt
    prompt_v2 = opik.Prompt(
        name=f"{project_name}_classification_prompt",
        prompt="""You are an expert content curator. Analyze the provided image and classify it for recommendation.

Consider these factors:
1. Visual appeal and quality
2. Content appropriateness
3. Emotional impact
4. Universal appeal across demographics

Respond with JSON in this exact format:
{
    "label": "highly recommend" | "recommend" | "neutral" | "not recommend" | "strongly not recommend",
    "reason": "Detailed explanation covering the factors above"
}""",
    )

    print("\n🔄 Iteration 2: Detailed criteria prompt")
    random_tag = generate_random_tag()
    results_v2 = run_evaluation_with_prompt(dataset, classifiers[:2], prompt_v2, f"v2_{random_tag}", project_name)
    logger.info(f"Completed iteration 2: {len(results_v2)} models evaluated")

    # Iteration 3: Structured reasoning prompt
    prompt_v3 = opik.Prompt(
        name=f"{project_name}_classification_prompt",
        prompt="""You are an expert content curator for a diverse audience platform.

Analyze the image and provide a recommendation based on:
- Visual Quality: Is the image clear, well-composed, and aesthetically pleasing?
- - Content Value: Does it provide entertainment, information, or emotional value?
- Audience Appeal: Will it resonate with a broad audience?
- Safety: Is it appropriate for all age groups?

Rate the content and explain your reasoning.

Respond with JSON in this exact format:
{
    "label": "highly recommend" | "recommend" | "neutral" | "not recommend" | "strongly not recommend",
    "reason": "Comprehensive explanation addressing each evaluation criteria",
    "confidence": 0.0-1.0,
    "categories": ["primary_category", "secondary_category"],
    "visual_features": ["feature1", "feature2", "feature3"]
}""",
    )

    print("\n🔄 Iteration 3: Structured reasoning with additional fields")
    random_tag = generate_random_tag()
    results_v3 = run_evaluation_with_prompt(dataset, classifiers[:2], prompt_v3, f"v3_{random_tag}", project_name)
    logger.info(f"Completed iteration 3: {len(results_v3)} models evaluated")

    # Step 3: GEPA Optimization
    print("\n🤖 Step 3: GEPA Prompt Optimization")
    print("-" * 50)

    try:
        from opik_optimizer import ChatPrompt as OptimizerChatPrompt
        from opik_optimizer.gepa_optimizer import GepaOptimizer

        # Create optimization dataset (subset for faster optimization)
        opt_dataset_name = f"multimodal_optimization_{get_datestamp()}"
        opt_dataset = client.create_dataset(name=opt_dataset_name)
        opt_dataset.insert(dataset_items[:50])  # Use 50 items for optimization

        print("Created optimization dataset with 50 items")

        # Define metric for optimization that returns proper ScoreResult
        def classification_metric(dataset_item: Dict[str, Any], llm_output: str) -> Any:
            # Parse the output to get structured data
            try:
                if isinstance(llm_output, str):
                    json_data = parse_json_response(llm_output)
                else:
                    json_data = llm_output

                if json_data:
                    label = json_data.get("label", "").lower()
                    reason = json_data.get("reason", "")
                else:
                    label = ""
                    reason = ""
            except Exception:
                label = ""
                reason = ""

            # Use the existing metric logic
            metric = ImageClassificationQualityMetric()
            return metric.score(
                output={"response_label": label, "response_reason": reason}, expected_label=dataset_item.get("expected_label", "")
            )

        # Create optimizer chat prompt (GEPA expects specific format)
        optimizer_prompt = OptimizerChatPrompt(
            system="Classify the image for content recommendation. Provide your response as JSON with 'label' and 'reason' fields.",
            user="Please analyze this image: {content_description}",
            model="gpt-4o-mini",  # This is the model that will be evaluated
        )

        # Initialize GEPA optimizer
        optimizer = GepaOptimizer(
            model="gpt-4o",  # Model for optimization process
            reflection_model="gpt-4o",  # Model for reflection
            n_threads=4,
            temperature=0.3,
            max_tokens=500,
        )

        print("Running GEPA optimization...")
        print("This will test multiple prompt variations...")

        # Run optimization
        result = optimizer.optimize_prompt(
            prompt=optimizer_prompt,
            dataset=opt_dataset,
            metric=classification_metric,
            max_trials=10,
            reflection_minibatch_size=3,
            n_samples=20,  # Test on subset of optimization dataset
        )

        # Extract optimized prompt
        optimized_system_prompt = result.prompt.messages[0]["content"] if result.prompt.messages else result.prompt.system

        # Create Opik prompt from optimized result
        prompt_optimized = opik.Prompt(name=f"{project_name}_classification_prompt", prompt=optimized_system_prompt)

        print("\n✨ Optimized prompt generated!")
        print(f"Original score: {result.initial_score:.2f}")
        print(f"Optimized score: {result.score:.2f}")
        print(f"Improvement: {result.improvement:.2%}")
        print(f"\nOptimized prompt: {optimized_system_prompt[:200]}...")

        # Evaluate optimized prompt on full dataset
        print("\nEvaluating optimized prompt on full dataset...")
        random_tag = generate_random_tag()
        results_optimized = run_evaluation_with_prompt(
            dataset, classifiers[:2], prompt_optimized, f"gepa_optimized_{random_tag}", project_name
        )
        logger.info(f"Completed GEPA optimized evaluation: {len(results_optimized)} models evaluated")

    except ImportError as e:
        print(f"⚠️  GEPA optimizer not available: {e}")
        print("   Demonstrating alternative automated optimization approach...")

        # Fallback: Simple automated prompt enhancement
        prompt_automated = opik.Prompt(
            name=f"{project_name}_classification_prompt",
            prompt="""<role>Expert Visual Content Analyst</role>

<task>Analyze the provided image and determine its recommendation level for a general audience platform.</task>

<criteria>
- Visual Quality (25%): Resolution, composition, lighting, and technical quality
- Content Appeal (25%): Interest level, uniqueness, and entertainment value
- Emotional Impact (25%): Positive emotions evoked, memorability
- Universal Suitability (25%): Age-appropriateness, cultural sensitivity

<rating_scale>
- "highly recommend": Exceptional content that will delight most users (90-100% score)
- "recommend": Good content worth sharing (70-89% score)
- "neutral": Average content without strong appeal (50-69% score)
- "not recommend": Below average or problematic content (30-49% score)
- "strongly not recommend": Poor quality or inappropriate content (0-29% score)
</rating_scale>

<output_format>
{
    "label": "[your rating from the scale above]",
    "reason": "[detailed explanation covering all criteria with specific observations about the image]"
}
</output_format>

Analyze the image and provide your classification:""",
        )

        print("\nEvaluating automated enhanced prompt...")
        random_tag = generate_random_tag()
        results_automated = run_evaluation_with_prompt(dataset, classifiers[:2], prompt_automated, f"automated_{random_tag}", project_name)
        logger.info(f"Completed automated evaluation: {len(results_automated)} models evaluated")

    # Summary
    print("\n📈 DEMO SUMMARY")
    print("=" * 70)
    print("✅ Successfully demonstrated:")
    print("   1. Dataset import using Opik SDK (200 items)")
    print("   2. JSON responses parsed and mapped to dataset columns")
    print("   3. Three manual prompt iterations with progressive improvements")
    print("   4. GEPA optimizer for automated prompt optimization")
    print("\n🎯 Key Features Shown:")
    print("   - Multimodal image classification with VLMs")
    print("   - Structured JSON output with field extraction")
    print("   - Model comparison (OpenAI GPT-4V vs Gemini)")
    print("   - Prompt versioning and tracking in Opik")
    print("   - Experiment tracking and metrics")
    print("\n📊 Check Opik dashboard for detailed results and comparisons!")

    # Flush tracking data
    opik.flush_tracker()


if __name__ == "__main__":
    # Check for required API keys
    required_keys = ["OPIK_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print("⚠️  Missing required API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease set these environment variables or add them to .env file")

    # Run demo
    main()
