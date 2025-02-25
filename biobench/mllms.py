import asyncio
import collections
import dataclasses
import logging
import time
import typing

import beartype
import litellm

from . import interfaces

logger = logging.getLogger("mllms")

# Disable logging for packages that yap a lot.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ExampleMllm:
    image_b64: str
    user: str
    assistant: str


@beartype.beartype
def fits(
    examples: list[Example],
    image_b64: str,
    user: str,
) -> bool:
    max_tokens = get_max_tokens(args)
    messages = make_prompt(args, examples, image_b64, user)
    n_tokens = litellm.token_counter(model=args.ckpt, messages=messages)
    return n_tokens <= max_tokens


@beartype.beartype
def get_max_tokens(args: interfaces.ModelArgsMllm) -> int:
    try:
        return litellm.get_max_tokens(args.ckpt)
    except Exception:
        pass

    if args.ckpt.endswith("google/gemini-2.0-flash-001"):
        return 1_000_000
    elif args.ckpt.endswith("google/gemini-flash-1.5-8b"):
        return 1_000_000
    elif args.ckpt.endswith("qwen/qwen-2-vl-7b-instruct"):
        return 4_096
    elif args.ckpt.endswith("meta-llama/llama-3.2-3b-instruct"):
        return 131_000
    else:
        err_msg = f"Model '{args.ckpt}' isn't mapped yet by biobench or litellm."
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
                if wait_time > 1.0:
                    logger.info("Sleeping for %.2f seconds.", wait_time)
                await asyncio.sleep(wait_time)

        # Add current timestamp
        self.timestamps.append(time.monotonic())


@beartype.beartype
async def send(
    args: interfaces.ModelArgsMllm,
    examples: list[Example],
    image_b64: str,
    user: str,
    *,
    max_retries: int = 5,
    system: str = "",
) -> str:
    """
    Send a message to the LLM and get the response.

    Args:
        args: Args for the MLLM.
        examples: Few-shot examples.
        image: The input image.
        user: The user request.

    Returns:
        The LLM's response as a string.

    Raises:
        ValueError: If required settings are missing
        RuntimeError: If LLM call fails
    """

    messages = make_prompt(args, examples, image_b64, user)

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
                temperature=args.temp,
                provider={"quantizations": args.quantizations},
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


def make_prompt(
    args: interfaces.ModelArgsMllm,
    examples: list[Example],
    image_b64: str,
    user: str,
    *,
    system: str = "",
) -> list[object]:
    if args.prompts == "single-turn":
        return _make_single_turn_prompt(examples, image_b64, user, system=system)
    elif args.prompt == "multi-turn":
        return _make_multi_turn_prompt(examples, image_b64, user, system=system)
    else:
        typing.assert_never(args.prompts)


def _make_single_turn_prompt(
    examples: list[Example], image_b64: str, user: str, *, system: str = ""
) -> list[object]:
    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    content = []
    for example in examples:
        content.append({"type": "image_url", "image_url": {"url": example.image_b64}})
        content.append({"type": "text", "text": f"{example.user}\n{example.assistant}"})

    content.append({"type": "image_url", "image_url": {"url": image_b64}})
    content.append({"type": "text", "text": user})

    messages.append({"role": "user", "content": content})

    return messages


def _make_multi_turn_prompt(
    examples: list[Example], image_b64: str, user: str, *, system: str = ""
) -> list[str]:
    # Format messages for chat completion
    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    for example in examples:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": example.image_b64}},
                {"type": "text", "text": example.user},
            ],
        })
        messages.append({"role": "assistant", "content": example.assistant})

    # Add current message
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_b64}},
            {"type": "text", "text": user},
        ],
    })
    return messages


class MultimodalLLM:
    """
    A minimal interface for interacting with multimodal language models.
    """

    def fits(self, examples: list[ExampleMllm], img_b64: str, user: str) -> bool:
        """
        Check if the given examples, image and prompt will fit in the model's context window.

        Args:
            TODO

        Returns:
            True if the inputs fit within the context window, False otherwise
        """
        max_tokens = self.get_max_tokens()
        messages = self.make_prompt(examples, image_b64, user)
        n_tokens = litellm.token_counter(model=args.ckpt, messages=messages)
        return n_tokens <= max_tokens
        err_msg = f"{self.__class__.__name__} must implemented fits()."
        raise NotImplementedError(err_msg)

    def get_max_tokens(self) -> int:
        """
        Get the maximum token limit for this model.

        Returns:
            Maximum token count this model can process
        """
        err_msg = f"{self.__class__.__name__} must implemented get_max_tokens()."
        raise NotImplementedError(err_msg)

    async def send(
        self,
        examples: list[ExampleMllm],
        image_b64: str,
        user: str,
        *,
        max_retries: int = 5,
        system: str = "",
        temperature: float = 0.0,
    ) -> str:
        """
        Send examples, image and prompt to the model and get a response.

        Args:
            TODO

        Returns:
            The model's text response
        """
        raise NotImplementedError()

    def make_prompt(
        self,
        args: interfaces.ModelArgsMllm,
        examples: list[Example],
        image_b64: str,
        user: str,
        *,
        system: str = "",
    ) -> list[object]:
        if args.prompts == "single-turn":
            return _make_single_turn_prompt(examples, image_b64, user, system=system)
        elif args.prompt == "multi-turn":
            return _make_multi_turn_prompt(examples, image_b64, user, system=system)
        else:
            typing.assert_never(args.prompts)

    def _make_single_turn_prompt(
        self, examples: list[Example], image_b64: str, user: str, *, system: str = ""
    ) -> list[object]:
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        content = []
        for example in examples:
            content.append({
                "type": "image_url",
                "image_url": {"url": example.image_b64},
            })
            content.append({
                "type": "text",
                "text": f"{example.user}\n{example.assistant}",
            })

        content.append({"type": "image_url", "image_url": {"url": image_b64}})
        content.append({"type": "text", "text": user})

        messages.append({"role": "user", "content": content})

        return messages

    def _make_multi_turn_prompt(
        self, examples: list[Example], image_b64: str, user: str, *, system: str = ""
    ) -> list[str]:
        # Format messages for chat completion
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        for example in examples:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": example.image_b64}},
                    {"type": "text", "text": example.user},
                ],
            })
            messages.append({"role": "assistant", "content": example.assistant})

        # Add current message
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_b64}},
                {"type": "text", "text": user},
            ],
        })
        return messages
