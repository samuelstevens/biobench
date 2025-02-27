import asyncio
import collections
import dataclasses
import logging
import time
import typing

import beartype
import litellm

from . import config

logger = logging.getLogger("mllms")

# Disable logging for packages that yap a lot.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


###################
# Data Structures #
###################


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Example:
    img_b64: str
    user: str
    assistant: str


@dataclasses.dataclass(frozen=True)
class Mllm:
    name: str
    max_tokens: int
    usd_per_m_input: float
    usd_per_m_output: float
    quantizations: list[str] = dataclasses.field(default_factory=list)


#############
# Functions #
#############


@beartype.beartype
def fits(
    cfg: config.Experiment, examples: list[Example], img_b64: str, user: str
) -> bool:
    mllm = load_mllm(cfg.model)
    messages = make_prompt(cfg, examples, img_b64, user)
    n_tokens = litellm.token_counter(model=cfg.model.ckpt, messages=messages)
    return n_tokens <= mllm.max_tokens


@beartype.beartype
async def send(
    cfg: config.Experiment,
    examples: list[Example],
    img_b64: str,
    user: str,
    *,
    max_retries: int = 5,
    system: str = "",
) -> str:
    """
    Send a message to the LLM and get the response.

    Args:
        cfg: TODO
        examples: Few-shot examples.
        image: The input image.
        user: The user request.

    Returns:
        The LLM's response as a string.

    Raises:
        ValueError: If required settings are missing
        RuntimeError: If LLM call fails
    """

    messages = make_prompt(cfg, examples, img_b64, user)
    mllm = load_mllm(cfg.model)

    # Make LLM call with retries
    last_err = None

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Exponential backoff: 2, 4, 8, 16, ... seconds
                wait_time_s = 2**attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %d seconds.",
                    attempt,
                    max_retries,
                    last_err,
                    wait_time_s,
                )
                await asyncio.sleep(wait_time_s)

            # Make LLM call
            response = await litellm.acompletion(
                model=cfg.model.org + "/" + cfg.model.ckpt,
                messages=messages,
                temperature=cfg.temp,
                provider={"quantizations": mllm.quantizations},
            )
        except RuntimeError as err:
            last_err = err
            if attempt == max_retries - 1:
                raise RuntimeError(f"Max retries ({max_retries}) exceeded: {err}")

        except litellm.APIConnectionError as err:
            should_retry = litellm._should_retry(err.status_code)
            if should_retry:
                last_err = err
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Max retries ({max_retries}) exceeded: {err}")
                continue
            raise RuntimeError(f"Non-retryable API connection error: {err}") from err

        except (litellm.APIError, litellm.BadRequestError) as err:
            # For some godforsaken reason, litellm does not parse the rate-limit error response from OpenRouter.
            # It just raises a litellm.APIError, which happens when it gets a bad response.
            # So in the interest of not failing, if err.llm_provider is 'openrouter' we assume it's a rate limit, and try again.
            if getattr(err, "llm_provider", None) == "openrouter":
                # Treat as rate limit for OpenRouter
                last_err = err
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Max retries ({max_retries}) exceeded: {err}")
                continue
            # For other providers, raise the error
            raise RuntimeError(f"API error: {err}") from err

        # Extract response and update history
        response = response.choices[0].message.content
        if response is None:
            return ""
        return response


#############
# PROMPTING #
#############


@beartype.beartype
def make_prompt(
    cfg: config.Experiment,
    examples: list[Example],
    img_b64: str,
    user: str,
    *,
    system: str = "",
) -> list[object]:
    if cfg.prompting == "single":
        return _make_single_turn_prompt(examples, img_b64, user, system=system)
    elif cfg.prompting == "multi":
        return _make_multi_turn_prompt(examples, img_b64, user, system=system)
    else:
        typing.assert_never(cfg.prompting)


@beartype.beartype
def _make_single_turn_prompt(
    examples: list[Example],
    img_b64: str,
    user: str,
    *,
    system: str = "",
) -> list[object]:
    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    content = []
    for example in examples:
        content.append({"type": "image_url", "image_url": {"url": example.img_b64}})
        content.append({"type": "text", "text": f"{example.user}\n{example.assistant}"})

    content.append({"type": "image_url", "image_url": {"url": img_b64}})
    content.append({"type": "text", "text": user})

    messages.append({"role": "user", "content": content})

    return messages


@beartype.beartype
def _make_multi_turn_prompt(
    examples: list[Example],
    img_b64: str,
    user: str,
    *,
    system: str = "",
) -> list[object]:
    # Format messages for chat completion
    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    for example in examples:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": example.img_b64}},
                {"type": "text", "text": example.user},
            ],
        })
        messages.append({"role": "assistant", "content": example.assistant})

    # Add current message
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": img_b64}},
            {"type": "text", "text": user},
        ],
    })
    return messages


###############
# PARALLELISM #
###############


@beartype.beartype
class RateLimiter:
    def __init__(self, max_rate: int, window_s: float = 1.0):
        self.max_rate = max_rate
        self.window_s = window_s
        self.timestamps = collections.deque()

    async def acquire(self):
        now = time.monotonic()

        # Remove timestamps older than our window
        while self.timestamps and now - self.timestamps[0] > self.window_s:
            self.timestamps.popleft()

        # If we're at max capacity, wait until oldest timestamp expires
        if len(self.timestamps) >= self.max_rate:
            wait_time = self.timestamps[0] + self.window_s - now
            if wait_time > 0:
                if wait_time > 1.0:
                    logger.info("Sleeping for %.2f seconds.", wait_time)
                await asyncio.sleep(wait_time)

        # Add current timestamp
        self.timestamps.append(time.monotonic())


############
# REGISTRY #
############


_global_mllm_registry: dict[tuple[str, str], Mllm] = {}


@beartype.beartype
def load_mllm(cfg: config.Model) -> Mllm:
    """
    Load a multimodal LLM configuration.
    """
    key = (cfg.org, cfg.ckpt)
    if key not in _global_mllm_registry:
        raise ValueError(f"Model '{key}' not found.")

    return _global_mllm_registry[key]


@beartype.beartype
def register_mllm(model_org: str, mllm: Mllm):
    """
    Register a new multimodal LLM configuration.
    """
    key = (model_org, mllm.name)
    if key in _global_mllm_registry:
        logger.warning("Overwriting key '%s' in registry.", key)
    _global_mllm_registry[key] = mllm


@beartype.beartype
def list_mllms() -> list[tuple[str, str]]:
    """
    List all registered multimodal LLM models.
    """
    return list(_global_mllm_registry.keys())


###################
# Built-in models #
###################

# Open-Source

register_mllm(
    "openrouter",
    Mllm("meta-llama/llama-3.2-3b-instruct", 131_000, 0.015, 0.025, ["fp32", "bf16"]),
)
register_mllm(
    "openrouter",
    Mllm("qwen/qwen-2-vl-7b-instruct", 4096, 0.1, 0.1, ["fp32", "bf16"]),
)

# Proprietary

register_mllm(
    "openrouter",
    Mllm("google/gemini-flash-1.5-8b", 1_000_000, 0.0375, 0.15),
)
register_mllm(
    "openrouter",
    Mllm("google/gemini-2.0-flash-lite-001", 1_000_000, 0.075, 0.3),
)
