"""Shared utilities for ofengtracker."""


def infer_provider(model: str) -> str:
    """Infer the provider from the model name.

    Args:
        model: Model name (e.g., 'gpt-4o-mini', 'gemini-2.0-flash-lite')

    Returns:
        Provider name ('openai' or 'gemini')
    """
    if model.startswith(("gpt-", "o1-", "o3-")):
        return "openai"
    elif model.startswith("gemini-"):
        return "gemini"
    else:
        # Default to OpenAI for unknown models
        return "openai"
