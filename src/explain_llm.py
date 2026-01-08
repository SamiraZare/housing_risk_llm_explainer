import os

def build_explanation_prompt(features: dict, regression_pred: float, risk_class: int, risk_proba: float) -> str:
    """
    Build a prompt to send to an LLM to explain the prediction.
    """
    lines = [
        "You are an AI assistant explaining housing price and rent risk predictions.",
        "Given the following house features and model outputs, explain the prediction",
        "in clear, non-technical language.",
        "",
        f"Features: {features}",
        f"Predicted price: {regression_pred:.2f}",
        f"Predicted rent risk (0 = low, 1 = high): {risk_class}",
        f"Predicted probability of high risk: {risk_proba:.2f}",
        "",
        "Explain why the model might make this prediction and what factors contribute most.",
    ]
    return "\n".join(lines)

def explain_with_llm(prompt: str) -> str:
    """
    Placeholder LLM call.
    Replace with actual API call (e.g., OpenAI) if desired.
    """
    # Example structure; fill in if you use OpenAI or similar.
    # import openai
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    # response = openai.ChatCompletion.create(
    #     model="gpt-4.1-mini",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response["choices"][0]["message"]["content"]

    # For now, just return the prompt so the pipeline is testable.
    return "LLM explanation placeholder.\n\nPROMPT:\n" + prompt
