import os
import opik
from opik_optimizer import (
    ChatPrompt,
    GepaOptimizer,
    MultiMetricObjective,
)
from opik.evaluation.metrics import score_result
from typing import Dict, Any, List
import time
import logging
from datetime import datetime


def create_text_dataset() -> List[Dict[str, Any]]:
    texts = [
        {"text": "URGENT! You've won $1000! Click here now to claim your prize!", "expected_label": "spam"},
        {"text": "Free money! No strings attached! Call now!", "expected_label": "spam"},
        {"text": "Congratulations! You're selected for our exclusive offer!", "expected_label": "spam"},
        {"text": "Special offer: 20% off all items this weekend only!", "expected_label": "spam"},
        {"text": "Don't miss out! Limited time deal - 50% off everything!", "expected_label": "spam"},
        {"text": "Exclusive discount for our valued customers!", "expected_label": "spam"},
        {"text": "Flash sale! Everything must go - up to 70% off!", "expected_label": "spam"},
        {"text": "Act now! Special discount for today only!", "expected_label": "spam"},
        {"text": "You won! Click to claim your prize immediately!", "expected_label": "spam"},
        {"text": "This is absolutely fantastic! Everyone should try it!", "expected_label": "spam"},
        {"text": "Thank you for your recent purchase. Your order has been shipped.", "expected_label": "legitimate"},
        {"text": "Meeting scheduled for tomorrow at 2 PM in conference room A.", "expected_label": "legitimate"},
        {"text": "Your monthly bank statement is now available online.", "expected_label": "legitimate"},
        {"text": "Reminder: Doctor's appointment next Tuesday at 10 AM.", "expected_label": "legitimate"},
        {"text": "New product launch! Check out our latest innovation.", "expected_label": "legitimate"},
        {"text": "We're excited to announce our new service offering.", "expected_label": "legitimate"},
        {"text": "Join us for our upcoming webinar next week.", "expected_label": "legitimate"},
        {"text": "The weather is nice today.", "expected_label": "legitimate"},
        {"text": "I'm not sure about this.", "expected_label": "legitimate"},
        {"text": "Maybe we should consider other options.", "expected_label": "legitimate"},
        {"text": "Great deal on our products this month!", "expected_label": "spam"},
        {"text": "Check out our amazing new features!", "expected_label": "spam"},
        {"text": "You'll love this incredible offer!", "expected_label": "spam"},
        {"text": "Don't miss this opportunity!", "expected_label": "spam"},
        {"text": "Limited time only - act fast!", "expected_label": "spam"},
        {"text": "Save big on our premium services!", "expected_label": "spam"},
        {"text": "Exclusive access for members only!", "expected_label": "spam"},
        {"text": "Best prices guaranteed!", "expected_label": "spam"},
        {"text": "Sign up now and get instant benefits!", "expected_label": "spam"},
        {"text": "Transform your life with our program!", "expected_label": "spam"},
        {"text": "We're pleased to announce our quarterly results.", "expected_label": "legitimate"},
        {"text": "Your account has been successfully updated.", "expected_label": "legitimate"},
        {"text": "Please review the attached document.", "expected_label": "legitimate"},
        {"text": "The meeting has been rescheduled to next week.", "expected_label": "legitimate"},
        {"text": "Your subscription will renew automatically.", "expected_label": "legitimate"},
        {"text": "We're launching a new feature next month.", "expected_label": "legitimate"},
        {"text": "Thank you for your feedback on our service.", "expected_label": "legitimate"},
        {"text": "Your order confirmation is attached.", "expected_label": "legitimate"},
        {"text": "We've updated our privacy policy.", "expected_label": "legitimate"},
        {"text": "Your payment has been processed successfully.", "expected_label": "legitimate"},
        {"text": "Click here for more information.", "expected_label": "spam"},
        {"text": "Reply STOP to unsubscribe.", "expected_label": "legitimate"},
        {"text": "Limited time offer expires soon!", "expected_label": "spam"},
        {"text": "Your trial period ends tomorrow.", "expected_label": "legitimate"},
        {"text": "Get rich quick with this method!", "expected_label": "spam"},
        {"text": "Your invoice is ready for download.", "expected_label": "legitimate"},
        {"text": "Act now before it's too late!", "expected_label": "spam"},
        {"text": "Your password has been reset.", "expected_label": "legitimate"},
        {"text": "Make money from home!", "expected_label": "spam"},
        {"text": "Your delivery is scheduled for today.", "expected_label": "legitimate"},
    ]
    return texts


def true_positive_metric(dataset_item: Dict[str, Any], llm_output: str) -> score_result.ScoreResult:
    predicted_label = llm_output.strip().lower()
    expected_label = dataset_item["expected_label"].lower()
    
    is_tp = expected_label == "spam" and predicted_label == "spam"
    value = 1.0 if is_tp else 0.0
    
    return score_result.ScoreResult(value=value, name="true_positive")


def true_negative_metric(dataset_item: Dict[str, Any], llm_output: str) -> score_result.ScoreResult:
    predicted_label = llm_output.strip().lower()
    expected_label = dataset_item["expected_label"].lower()
    
    is_tn = expected_label == "legitimate" and predicted_label == "legitimate"
    value = 1.0 if is_tn else 0.0
    
    return score_result.ScoreResult(value=value, name="true_negative")


def main():
    os.environ["OPIK_API_KEY"] = os.getenv("OPIK_API_KEY")
    os.environ["OPIK_WORKSPACE"] = os.getenv("OPIK_WORKSPACE_NAME")
    os.environ["OPIK_PROJECT_NAME"] = "Text-Classification-GEPA-Demo"
    
    # Configure tracing to prevent timeout errors
    os.environ["OPIK_REQUEST_TIMEOUT"] = "300"  # 5 minutes timeout for large payloads
    os.environ["OPIK_BATCH_SIZE"] = "10"  # Smaller batch size to prevent timeouts
    
    # Suppress LiteLLM OpikLogger timeout errors (fallback)
    logging.getLogger("litellm.integrations.opik").setLevel(logging.WARNING)

    client = opik.Opik()
    dataset_items = create_text_dataset()
    
    timestamp = int(time.time())
    dataset_name = f"text_classification_gepa_demo_{timestamp}"
    dataset = client.create_dataset(name=dataset_name)
    dataset.insert(dataset_items)
    
    prompt = ChatPrompt(
        system="""You are a spam detection expert. Analyze the provided text and classify it as either spam or legitimate.

        Respond with only one word: either "spam" or "legitimate".
        """,
        user="{text}",
    )
    
    multi_metric_objective = MultiMetricObjective(
        weights=[1.0, 1.0],
        metrics=[true_positive_metric, true_negative_metric],
        name="maximize_tp_tn_weighted",
    )
    
    optimizer = GepaOptimizer(
        model="openai/gpt-3.5-turbo",
    )
    
    result = optimizer.optimize_prompt(
        prompt=prompt,
        dataset=dataset,
        metric=multi_metric_objective,
        n_samples=20,
    )
    
    result.display()
    opik.flush_tracker()
    
    # Save best prompt to prompt library
    system_prompt = result.prompt[0].get("content")
    
    score = result.score
    
    metadata = {
        "optimization_score": float(score) if score else None,
        "optimization_date": str(datetime.now()),
        "optimizer": "GEPA",
        "model": "openai/gpt-3.5-turbo",
    }
    
    client.create_prompt(
        name="spam_detection_optimized",
        prompt=system_prompt,
        metadata=metadata
    )


if __name__ == "__main__":
    main()