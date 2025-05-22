# -*- coding: utf-8 -*-
"""
opik_call_summarizer_langgraph.py

This script demonstrates using Opik with LangGraph to build a call transcript
analyzer. It classifies a call, then summarizes it and extracts action items
based on the call type.

Setup Required:
1. Install necessary libraries:
   pip install comet_llm langgraph langchain_openai langchain ipython

2. Set Environment Variables:
   - OPENAI_API_KEY: Your OpenAI API key.
   - COMET_API_KEY: Your Comet API key.
   - COMET_WORKSPACE: Your Comet workspace name.
   - COMET_PROJECT_NAME: The Comet project where traces will be logged (e.g., "opik-langgraph-demo").

You can get your Comet API key from your Comet account settings.
"""

import os
from typing import TypedDict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph

import opik
from opik.integrations.langchain import OpikTracer
from dotenv import load_dotenv

load_dotenv()

opik.configure(
    use_local=False,
    workspace=os.getenv("COMET_WORKSPACE"),
    api_key=os.getenv("COMET_API_KEY")
)

os.environ["OPIK_PROJECT_NAME"] = os.getenv("COMET_PROJECT_NAME")

# --- Configuration ---
# Ensure your OpenAI API key is set
# os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY" # Or set as an environment variable
if not os.getenv("ANTHROPIC_API_KEY"):
    print(
        "Error: ANTHROPIC_API_KEY not set. "
        "Please set it as an environment variable or in the script."
    )
    # exit() # Uncomment to exit if key is not found

# Ensure Comet variables are set for Opik tracing
# os.environ["COMET_API_KEY"] = "YOUR_COMET_API_KEY"
# os.environ["COMET_WORKSPACE"] = "YOUR_COMET_WORKSPACE"
# os.environ["COMET_PROJECT_NAME"] = "demo-langgraph-callsummarizer" # Choose your project name
if not all(os.getenv(var) for var in ["COMET_API_KEY", "COMET_WORKSPACE", "COMET_PROJECT_NAME"]):
    print(
        "Warning: COMET_API_KEY, COMET_WORKSPACE, or COMET_PROJECT_NAME not set. "
        "Opik tracing will not work without them. Please set them as environment variables."
    )


# --- State Definition ---
class CallAnalysisState(TypedDict):
    transcript: str
    call_type: str
    summary: Optional[str]
    action_items: Optional[List[str]]
    raw_llm_output: Optional[str]  # Stores the direct output from the summarizer LLM
    error_message: Optional[str]


# --- LLM Initialization ---
# Using claude-3-sonnet-20240229 as the model
try:
    llm = ChatAnthropic(model="claude-3-sonnet-20240229")
except Exception as e:
    print(f"Failed to initialize LLM. Have you set your ANTHROPIC_API_KEY? Error: {e}")
    llm = None # Allow script to load but operations will fail.

# --- Node Functions ---

def classify_call_type_node(state: CallAnalysisState):
    """Classifies the call transcript into predefined categories."""
    if not llm:
        return {"call_type": "unknown", "error_message": "LLM not initialized."}
    transcript = state["transcript"]
    prompt = f"""Given the following call transcript, classify it into one of these categories: sales, tech_support, check_in_update.
Return ONLY the category name in lowercase (e.g., sales, tech_support, check_in_update).

Transcript:
{transcript}

Category:"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        classification = response.content.strip().lower()
        if classification not in ["sales", "tech_support", "check_in_update"]:
            # If the LLM returns something unexpected, default to "unknown"
            print(f"Warning: LLM returned unexpected classification '{classification}'. Defaulting to 'unknown'.")
            classification = "unknown"
        return {"call_type": classification, "error_message": None}
    except Exception as e:
        print(f"Error during classification: {e}")
        return {"call_type": "unknown", "error_message": f"Classification failed: {str(e)}"}


def _generate_and_parse_summary(transcript: str, prompt_template: str, llm_instance: ChatAnthropic):
    """Helper function to generate and parse summary and action items."""
    if not llm_instance:
         return {
            "summary": "LLM not initialized.",
            "action_items": [],
            "raw_llm_output": "LLM not initialized.",
            "error_message": "LLM not initialized."
        }
    formatted_prompt = prompt_template.format(transcript=transcript)
    try:
        response = llm_instance.invoke([HumanMessage(content=formatted_prompt)])
        raw_output = response.content.strip()

        summary_part = ""
        action_items_list = []

        summary_marker = "Summary:"
        action_items_marker = "Action Items:"

        summary_start_index = raw_output.find(summary_marker)
        action_items_start_index = raw_output.find(action_items_marker)

        if summary_start_index != -1:
            if action_items_start_index != -1 and action_items_start_index > summary_start_index:
                summary_part = raw_output[summary_start_index + len(summary_marker):action_items_start_index].strip()
                action_items_block = raw_output[action_items_start_index + len(action_items_marker):].strip()
            else:
                summary_part = raw_output[summary_start_index + len(summary_marker):].strip()
                action_items_block = ""
        elif action_items_start_index != -1: # Only action items marker found
            action_items_block = raw_output[action_items_start_index + len(action_items_marker):].strip()
            # Potentially treat content before action items as summary, or leave summary_part empty
            # For simplicity, let's assume summary should have its marker.
        else: # Neither marker found
            summary_part = raw_output # Treat the whole output as summary
            action_items_block = ""
            print("Warning: Could not find 'Summary:' or 'Action Items:' markers in LLM output. Full output treated as summary.")


        if action_items_block:
            action_items_list = [
                item.strip().lstrip('- ')
                for item in action_items_block.split('\n')
                if item.strip() and item.strip().lstrip('- ')
            ]
        elif not action_items_list and action_items_start_index != -1 : # Action items marker was present but block was empty or parsing failed.
            print("Warning: 'Action Items:' marker found, but no action items were parsed.")


        return {
            "summary": summary_part,
            "action_items": action_items_list,
            "raw_llm_output": raw_output,
            "error_message": None
        }
    except Exception as e:
        print(f"Error during summary generation: {e}")
        return {
            "summary": f"Summary generation failed: {str(e)}",
            "action_items": [],
            "raw_llm_output": f"Error: {str(e)}",
            "error_message": f"LLM call for summarization failed: {str(e)}"
        }


def process_sales_node(state: CallAnalysisState):
    transcript = state["transcript"]
    prompt_template = """You are a helpful assistant. Based on the following sales call transcript, provide a concise summary and a list of action items.
Focus on customer needs, product interest, potential deal size, and next steps for the sales team.

Transcript:
{transcript}

Respond in the following format AND NOTHING ELSE:
Summary: <Your concise summary here>
Action Items:
- <Action item 1>
- <Action item 2>"""
    return _generate_and_parse_summary(transcript, prompt_template, llm)


def process_tech_support_node(state: CallAnalysisState):
    transcript = state["transcript"]
    prompt_template = """You are a helpful assistant. Based on the following technical support call transcript, provide a concise summary of the issue, the troubleshooting steps taken, the resolution (if any), and a list of action items.

Transcript:
{transcript}

Respond in the following format AND NOTHING ELSE:
Summary: <Your concise summary here, including issue, steps, and resolution>
Action Items:
- <Action item 1 for support agent>
- <Action item 2 for customer, if any>"""
    return _generate_and_parse_summary(transcript, prompt_template, llm)


def process_check_in_node(state: CallAnalysisState):
    transcript = state["transcript"]
    prompt_template = """You are a helpful assistant. Based on the following check-in/update call transcript, provide a concise summary of the discussion, key updates shared, any concerns raised, and a list of action items.

Transcript:
{transcript}

Respond in the following format AND NOTHING ELSE:
Summary: <Your concise summary here, including updates and concerns>
Action Items:
- <Action item 1>
- <Action item 2>"""
    return _generate_and_parse_summary(transcript, prompt_template, llm)


def handle_unknown_node(state: CallAnalysisState):
    return {
        "summary": "Call type is 'unknown'. Could not process with a specific template.",
        "action_items": [],
        "raw_llm_output": f"Classification result: {state.get('call_type', 'N/A')}.",
        "error_message": state.get("error_message") or "Call classified as 'unknown' or classification failed."
    }


# --- Conditional Router ---
def route_by_call_type(state: CallAnalysisState):
    call_type = state["call_type"]
    # If classification itself had an error leading to 'unknown'
    if state.get("error_message") and "Classification failed" in state.get("error_message", ""):
        return "handle_unknown"

    if call_type == "sales":
        return "process_sales"
    elif call_type == "tech_support":
        return "process_tech_support"
    elif call_type == "check_in_update":
        return "process_check_in"
    else:  # "unknown" or any other unexpected value from classification
        return "handle_unknown"


# --- Graph Definition ---
workflow = StateGraph(CallAnalysisState)

workflow.add_node("classify_call_type", classify_call_type_node)
workflow.add_node("process_sales", process_sales_node)
workflow.add_node("process_tech_support", process_tech_support_node)
workflow.add_node("process_check_in", process_check_in_node)
workflow.add_node("handle_unknown", handle_unknown_node)

workflow.set_entry_point("classify_call_type")

workflow.add_conditional_edges(
    "classify_call_type",
    route_by_call_type,
    {
        "process_sales": "process_sales",
        "process_tech_support": "process_tech_support",
        "process_check_in": "process_check_in",
        "handle_unknown": "handle_unknown",
    }
)

workflow.add_edge("process_sales", END)
workflow.add_edge("process_tech_support", END)
workflow.add_edge("process_check_in", END)
workflow.add_edge("handle_unknown", END)

app = workflow.compile()


# --- Example Transcripts ---
example_transcript_sales = """
Agent: Good morning! Thank you for calling FutureTech Solutions, this is Alex speaking. How can I help you today?
Customer: Hi Alex, I'm interested in your new AI-powered CRM. We're a growing e-commerce business and our current system is just not keeping up.
Agent: Absolutely! Our AI CRM is perfect for scaling businesses. It offers predictive analytics, automated customer segmentation, and personalized marketing campaign tools. Could you tell me a bit more about your current challenges and what you're looking for?
Customer: We primarily struggle with managing a large volume of customer inquiries and personalizing outreach. We also want better sales forecasting. Our team is about 50 sales reps.
Agent: I see. Our system handles high volumes seamlessly and the AI can draft personalized responses for reps to review. The forecasting module is also very accurate, typically within 5% of actuals. For a team of 50, we have an enterprise plan that would be a great fit. Would you be interested in a detailed demo next week? I can also send over a pricing sheet.
Customer: A demo sounds great. Yes, please send the pricing. Could we aim for Tuesday morning?
Agent: Tuesday morning works perfectly. I'll send a calendar invite shortly with the demo link and the pricing information. Is there anything else I can assist with?
Customer: No, that's all for now. Thanks, Alex!
Agent: You're welcome! Looking forward to our demo. Have a great day!
"""

example_transcript_tech_support = """
Customer: Hi, I'm having trouble with my internet connection. It keeps dropping every few minutes.
Agent: Hello, I'm sorry to hear you're experiencing issues. This is Sarah from Tech Support. Can I get your account number or phone number to pull up your details?
Customer: Sure, it's 555-0123.
Agent: Thank you. I see your account. Let's try a few troubleshooting steps. Have you tried restarting your modem and router?
Customer: Yes, I did that twice. It helps for about 10 minutes, then it starts dropping again.
Agent: Okay. Could you check the lights on your modem? Are any of them blinking or a different color than usual, like red or orange?
Customer: Hmm, let me see... Yes, the 'Internet' light is blinking orange. It's usually solid green.
Agent: Alright, an orange blinking light usually indicates a signal issue reaching the modem. There might be an outage in your area or a problem with the line. Let me check for outages... It seems there's some localized maintenance happening that might be affecting your service. It's scheduled to be completed in about 2 hours.
Customer: Oh, I see. So I just have to wait?
Agent: Yes, unfortunately, that's the case for now. If the issue persists after 2-3 hours, please call us back, and we can investigate further, possibly scheduling a technician. I'll also put a note on your account to monitor the connection from our end once the maintenance window is over.
Customer: Okay, thank you for the information, Sarah.
Agent: You're welcome! Is there anything else I can help you with today?
Customer: No, that's it.
Agent: Alright, have a good day, and I hope your connection is stable soon.
"""

example_transcript_check_in = """
Manager: Hi Ben, thanks for making time for our weekly check-in. How's Project Alpha progressing?
Ben: Hi Sarah. Project Alpha is on track. We completed the user testing phase for the new dashboard this week. Feedback was generally positive, with a few minor UI tweaks suggested.
Manager: That's great to hear. What were the key suggestions for the UI?
Ben: Mostly around the color contrast for accessibility and simplifying the date filter. The development team is already working on those. We expect to deploy the changes by end of day tomorrow.
Manager: Excellent. Any roadblocks or concerns with the upcoming deliverables for next week? We have the client presentation on Wednesday.
Ben: No major roadblocks. The report generation module is taking a bit longer than anticipated due to some complex data integrations, but we've allocated an extra resource, and I'm confident we'll have it ready for the presentation. The slides are also 80% complete.
Manager: Good to know you're managing the report module. Let's sync up on Monday afternoon for a quick review of the presentation draft.
Ben: Sounds good. I'll send you the draft by Monday noon.
Manager: Perfect. Anything else you need from my end or any other updates?
Ben: No, I think that covers it for Project Alpha. On a side note, I'm also making good progress on the Q3 planning document you asked for. Should have a first draft by Friday.
Manager: Fantastic, Ben. Thanks for the thorough update. Keep up the great work!
Ben: Thanks, Sarah! Talk to you on Monday.
"""

example_transcript_ambiguous = """
Person A: Hey, did you get that email I sent over about the TPS reports?
Person B: Oh, hi! Yeah, I saw it come in. Listen, I'm running to another meeting, but it looked important.
Person A: It is, we need to get those finalized by EOD. Also, the system was acting weird this morning when I tried to log my hours.
Person B: Okay, make a note of the system issue. I gotta run, let's circle back later!
"""

# --- Running the graph with Opik tracing ---
def run_analysis(transcript: str, transcript_id: str):
    if not llm:
        print("LLM not initialized. Cannot run analysis.")
        return None
    # Check for required environment variables
    required_vars = ["COMET_API_KEY", "COMET_WORKSPACE", "COMET_PROJECT_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Warning: Missing required environment variables: {', '.join(missing_vars)}")
        print("Running without Opik tracing. To enable tracing, please set the missing environment variables.")
        tracer_callback = None
    else:
        try:
            # Initialize the tracer with the graph and metadata
            tracer_callback = OpikTracer(
                graph=app.get_graph(xray=True),
                metadata={"transcript_id": transcript_id},
            )
            print(f"OpikTracer initialized. Trace will be logged to project '{os.getenv('COMET_PROJECT_NAME')}' with ID '{transcript_id}'.")
        except Exception as e:
            print(f"Error initializing OpikTracer: {str(e)}")
            print("Running without Opik tracing.")
            tracer_callback = None


    inputs = {"transcript": transcript}
    config = {"callbacks": [tracer_callback]} if tracer_callback else {}

    print(f"\n--- Analyzing Transcript ID: {transcript_id} ---")
    result = app.invoke(inputs, config=config)

    print("\n--- Analysis Result ---")
    print(f"Call Type: {result.get('call_type')}")
    print(f"Summary: {result.get('summary')}")
    print("Action Items:")
    if result.get('action_items'):
        for item in result.get('action_items'):
            print(f"- {item}")
    else:
        print("  (No action items extracted)")
    if result.get('error_message'):
        print(f"Error Message: {result.get('error_message')}")
    # print(f"Raw LLM Output for Summarizer: {result.get('raw_llm_output')}") # Uncomment for debugging

    if tracer_callback:
        print(f"\nTrace for '{transcript_id}' logged to Comet Opik.")
        print("You can view it in your Comet project under the Traces tab.")
    return result

if __name__ == "__main__":
    # Make sure API keys are set before running!
    if os.getenv("ANTHROPIC_API_KEY") and llm:
        run_analysis(example_transcript_sales, "sales-example-001")
        run_analysis(example_transcript_tech_support, "tech-support-example-001")
        run_analysis(example_transcript_check_in, "check-in-example-001")
        run_analysis(example_transcript_ambiguous, "ambiguous-example-001") # Test unknown/fallback
    else:
        print("\nSkipping analysis runs as OPENAI_API_KEY is not set or LLM failed to initialize.")