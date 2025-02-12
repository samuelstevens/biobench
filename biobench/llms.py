import asyncio
import base64
import collections
import dataclasses
import io
import logging
import time

import beartype
import litellm
from PIL import Image

from . import interfaces

logger = logging.getLogger("vlms")

# Disable logging for packages that yap a lot.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Example:
    image: Image.Image
    user: str
    assistant: str

    def to_history(self) -> list[object]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_base64(self.image)},
                    },
                ],
            },
            {"role": "assistant", "content": self.assistant},
        ]


@beartype.beartype
def fits(
    args: interfaces.ModelArgsVlm,
    examples: list[Example],
    image: Image.Image,
    user: str,
) -> bool:
    max_tokens = get_max_tokens(args)
    messages = []
    for example in examples:
        messages.extend(example.to_history())
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user},
            {
                "type": "image_url",
                "image_url": {"url": image_to_base64(image)},
            },
        ],
    })
    n_tokens = litellm.token_counter(model=args.ckpt, messages=messages)
    return n_tokens <= max_tokens


@beartype.beartype
def get_max_tokens(args: interfaces.ModelArgsVlm) -> int:
    try:
        return litellm.get_max_tokens(args.ckpt)
    except Exception:
        pass

    if args.ckpt.endswith("google/gemini-2.0-flash-001"):
        return 1_000_000
    if args.ckpt.endswith("google/gemini-flash-1.5-8b"):
        return 1_000_000
    else:
        err_msg = f"Model {args.ckpt} isn't mapped yet by biobench or litellm."
        raise ValueError(err_msg)


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
                logger.info("Sleeping for %.2f seconds.", wait_time)
                await asyncio.sleep(wait_time)

        # Add current timestamp
        self.timestamps.append(time.monotonic())


@beartype.beartype
async def send(
    args: interfaces.ModelArgsVlm,
    examples: list[Example],
    image: Image.Image,
    user: str,
    *,
    max_retries: int = 5,
) -> str:
    """
    Send a message to the LLM and get the response.

    Args:
        args: Args for the VLM.
        examples: Few-shot examples.
        image: The input image.
        user: The user request.

    Returns:
        The LLM's response as a string.

    Raises:
        ValueError: If required settings are missing
        RuntimeError: If LLM call fails
    """

    # Get optional settings with defaults.
    temperature = 0.7 if args.temp is None else args.temp

    # Format messages for chat completion
    messages = []
    for example in examples:
        messages.extend(example.to_history())

    # Add current message
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user},
            {
                "type": "image_url",
                "image_url": {"url": image_to_base64(image)},
            },
        ],
    })
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
                model=args.ckpt,
                messages=messages,
                temperature=temperature,
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
        return response


@beartype.beartype
def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="webp")
    b64 = base64.b64encode(buf.getvalue())
    s64 = b64.decode("utf8")
    return "data:image/webp;base64," + s64
