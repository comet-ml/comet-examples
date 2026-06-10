# Multimodal Image Classification Demo with Opik

This demo showcases Opik's capabilities for evaluating multimodal AI models on image classification tasks.

## Features Demonstrated

1. **SDK-based Dataset Import**: Create and import a 200-item synthetic dataset using the Opik SDK
2. **Prompt Iterations**: Progressive prompt improvements with JSON response mapping to dataset columns
3. **GEPA Optimization**: Automated prompt optimization using reflection-driven search for improved classification performance

## Prerequisites

### Install Dependencies

```bash
poetry install
```

### Set Up API Keys

Create a `.env` file with your API keys:

```bash
# Opik/Comet Configuration
COMET_API_KEY=your-comet-api-key-here
OPIK_WORKSPACE_NAME=your-workspace-name

# Model Provider API Keys
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
OPENROUTER_API_KEY=your-openrouter-api-key-here  # Optional for Qwen-VL
```

## Running the Demo

```bash
python multimodal_classification_demo.py
```

## What the Demo Does

1. **Dataset Creation**:
   - Downloads images from public Unsplash URLs
   - Creates a 200-item synthetic dataset
   - Imports into Opik with proper schema

2. **Model Comparison**:
   - OpenAI GPT-4V (Vision)
   - Google Gemini 1.5 Pro
   - Qwen-VL (via OpenRouter) - optional

3. **Prompt Iterations**:
   - **v1**: Basic classification prompt
   - **v2**: Detailed criteria with factors
   - **v3**: Structured reasoning with additional JSON fields

4. **JSON Field Extraction**:
   - Parses model responses
   - Maps fields to dataset columns:
     - `response_label`: Classification label
     - `response_reason`: Explanation
     - `response_confidence`: Confidence score (when available)
     - `response_categories`: Content categories
     - `response_visual_features`: Visual features identified

5. **GEPA Optimization**:
   - Uses reflection-driven evolutionary search
   - Runs on 50-item subset for speed
   - Generates optimized prompts automatically
   - Evaluates on full dataset

## Expected Output

The demo will:
- Create a multimodal dataset in your Opik workspace
- Run multiple experiments with different prompts
- Show progressive improvements in classification quality
- Display results in the Opik dashboard

## Viewing Results

1. Log into your Opik dashboard
2. Navigate to the `multimodal-classification-demo` project
3. Compare experiments across different prompt versions
4. Analyze the extracted JSON fields in dataset columns

## Customization

To add more images or change classification criteria:
1. Edit `utils.py` - `create_synthetic_image_data()` function
2. Modify the prompt templates in the main script
3. Adjust the metric scoring in `ImageClassificationQualityMetric`

## Troubleshooting

- **Import errors**: Ensure all dependencies are installed via Poetry
- **API key errors**: Check your `.env` file has all required keys
- **Image download failures**: The code includes fallbacks for failed downloads
- **JSON parsing errors**: The parser handles various response formats automatically
