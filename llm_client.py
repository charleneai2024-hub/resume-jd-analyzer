"""Reusable LLM client wrapper: retry with exponential backoff + cost logging."""

import random
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

# USD per 1M tokens: (input_price, output_price)
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}

MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0

# max_retries=0: we own the retry loop below instead of the SDK's built-in one.
_client = anthropic.Anthropic(max_retries=0)


def _is_retryable(error: anthropic.APIStatusError) -> bool:
    """Rate limits and server errors are transient; everything else (bad request, auth, permission, not found) will fail the same way every time."""
    return isinstance(error, (anthropic.RateLimitError, anthropic.InternalServerError))


def _log_cost(model: str, usage) -> None:
    input_price, output_price = MODEL_PRICING.get(model, (0.0, 0.0))
    cost = (usage.input_tokens / 1_000_000) * input_price + (usage.output_tokens / 1_000_000) * output_price
    print(
        f"[llm_client] model={model} input_tokens={usage.input_tokens} "
        f"output_tokens={usage.output_tokens} cost=${cost:.6f}"
    )


def _create_with_retry(**kwargs) -> anthropic.types.Message:
    """Core retry loop: send a messages.create request with exponential
    backoff on rate limits, server errors, and connection errors."""
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = _client.messages.create(**kwargs)
            _log_cost(kwargs["model"], response.usage)
            return response
        except anthropic.APIStatusError as error:
            if not _is_retryable(error):
                raise
            last_error = error
        except anthropic.APIConnectionError as error:
            last_error = error

        delay = BASE_DELAY_SECONDS * (2**attempt) + random.uniform(0, 1)
        print(
            f"[llm_client] retryable error ({last_error}); "
            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
        )
        time.sleep(delay)

    raise last_error


def call_llm(prompt: str, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1024) -> str:
    """Send one prompt to Claude and return the text response. (Original
    single-prompt interface, kept for backward compatibility.)"""
    response = _create_with_retry(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_llm_messages(
    messages: list[dict],
    system: str = "",
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Send a full conversation (with optional system prompt) to Claude.

    `messages` is a list of {"role": "user"/"assistant", "content": str}.
    System text goes in the separate `system` parameter — Anthropic's API
    keeps it outside the messages list, unlike OpenAI's role-based style.
    """
    response = _create_with_retry(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        temperature=temperature,
    )
    return response.content[0].text
