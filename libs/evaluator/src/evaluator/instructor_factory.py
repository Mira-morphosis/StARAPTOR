import instructor
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Dict, List, Optional

load_dotenv()
provider_type = os.getenv("STARAPTOR_PROVIDER_TYPE", "openai_compatible").lower()
raw_base_url = os.getenv("STARAPTOR_BASE_URL") or None
# Normalized once, with a guaranteed trailing slash: both "<base>v1" (the
# OpenAI-compatible API surface) and "<base>tokenize" (llama.cpp endpoint)
# then resolve correctly whether or not the environment variable ends with "/".
base_url = (raw_base_url.rstrip("/") + "/") if raw_base_url else None
api_key = os.getenv("STARAPTOR_API_KEY", "staraptor")
mode_str = os.getenv("STARAPTOR_INSTRUCTOR_MODE", "JSON")


v1_url = base_url + "v1" if base_url is not None else None


def get_request_extras() -> Dict[str, Dict]:
    """
    Returns the provider-specific request parameters shared by every extraction
    call, so the decoding protocol (temperature 0, reasoning disabled) is declared
    in a single place and stays uniform across the production pipeline and the
    benchmarks:
      * llama.cpp server (openai_compatible): thinking disabled via the chat-template
        kwarg, with a zero reasoning budget as a second guard;
      * Anthropic: adaptive thinking disabled;
      * other providers (e.g., OpenAI): no extras yet; reasoning-effort fields must
        be added here when the provider is integrated.
    :return: The dict to pass as extra_body in every chat.completions call.
    """
    if provider_type == "openai_compatible":
        return {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_budget": 0,
        }
    elif provider_type == "anthropic":
        return {"thinking": {"type": "disabled"}}
    else:
        return {}


def get_decoding_metadata() -> dict:
    """
    Gathers the decoding/runtime configuration actually used by the extraction calls,
    ready to be embedded in benchmark reports (methodology: decoding notes).
    :return: A dict describing provider, temperature, request extras and concurrency.
    """
    return {
        "provider_type": provider_type,
        "base_url": base_url,
        "instructor_mode": mode_str.upper(),
        "temperature": 0.0,
        "request_extras": get_request_extras(),
        "concurrency_limit": int(os.getenv("STARAPTOR_CONCURRENCY_LIMIT", "100")),
    }


def get_instructor_client():
    """
    Builds an Instructor-wrapped LLM client based on the configured provider.
    :return: An Instructor client (OpenAI-compatible or Anthropic backend) ready for structured-output calls.
    """

    # Instructor requires an enum, so we convert the string
    mode = getattr(instructor.Mode, mode_str.upper(), instructor.Mode.JSON)

    if provider_type == "openai_compatible":
        from openai import AsyncOpenAI
        return instructor.from_openai(
            AsyncOpenAI(base_url=v1_url, api_key=api_key),
            mode=mode
        )
    elif provider_type == "anthropic":
        from anthropic import AsyncAnthropic
        return instructor.from_anthropic(
            AsyncAnthropic(api_key=api_key)
        )
    else:
        raise ValueError(f"Provider '{provider_type}' is not supported.")


async def get_available_models() -> List[str]:
    """
    Lists models available from the configured provider, falling back to a static list if unsupported.
    :return: A list of model identifiers, either queried live via '/models' or read from the fallback env var.
    """
    fallback_models = [m.strip() for m in os.getenv("LLM_AVAILABLE_MODELS_FALLBACK", "").split(",") if m.strip()]

    if provider_type != "openai_compatible" or not base_url:
        return fallback_models

    # noinspection PyBroadException
    try:
        from openai import AsyncOpenAI
        async with AsyncOpenAI(base_url=v1_url, api_key=api_key) as raw_client:
            answer = await raw_client.models.list()
            return [model.id for model in answer.data]
    except Exception:
        return fallback_models


# ---------------------------------------------------------------------------
# Extra (e.g. frontier/cloud) providers.
#
# These are NEVER read from .env: API keys must not sit in a file next to the repo.
# They are supplied at execution time (e.g. as CLI arguments to the RQ1 benchmark script),
# parsed into ExtraProvider objects, and used only for that run.
# ---------------------------------------------------------------------------

@dataclass
class ExtraProvider:
    """An additional LLM provider (e.g. a frontier cloud API) supplied at runtime."""
    name: str
    type: str                                  # "openai_compatible" | "anthropic"
    api_key: str
    base_url: Optional[str] = None             # None = provider's default cloud endpoint
    models: List[str] = field(default_factory=list)  # empty = list live from the endpoint


def _extra_v1_url(spec: ExtraProvider) -> Optional[str]:
    """Returns the /v1 URL for an openai_compatible extra provider, or None for the cloud default."""
    return spec.base_url.rstrip("/") + "/v1" if spec.base_url else None


def get_client_for_extra(spec: ExtraProvider, mode_str: str = "JSON"):
    """
    Builds an Instructor-wrapped client for an extra provider.
    :param spec: The extra-provider configuration (from CLI, not env).
    :param mode_str: The Instructor mode (JSON, TOOLS, ...).
    :return: An Instructor client (OpenAI-compatible or Anthropic backend).
    """
    mode = getattr(instructor.Mode, mode_str.upper(), instructor.Mode.JSON)

    if spec.type == "openai_compatible":
        from openai import AsyncOpenAI
        return instructor.from_openai(
            AsyncOpenAI(base_url=_extra_v1_url(spec), api_key=spec.api_key),
            mode=mode
        )
    elif spec.type == "anthropic":
        from anthropic import AsyncAnthropic
        return instructor.from_anthropic(
            AsyncAnthropic(api_key=spec.api_key),
            mode=mode
        )
    else:
        raise ValueError(f"Extra provider type '{spec.type}' is not supported.")


async def list_models_for_extra(spec: ExtraProvider) -> List[str]:
    """
    Resolves the model ids of an extra provider: the explicit `models` list if given,
    otherwise a live query of the endpoint's /models (openai_compatible only).
    :param spec: The extra-provider configuration.
    :return: The list of model identifiers to benchmark.
    """
    if spec.models:
        return spec.models
    if spec.type == "openai_compatible":
        from openai import AsyncOpenAI
        async with AsyncOpenAI(base_url=_extra_v1_url(spec), api_key=spec.api_key) as raw_client:
            answer = await raw_client.models.list()
            return [model.id for model in answer.data]
    raise ValueError(
        f"Extra provider '{spec.name}' ({spec.type}) cannot list models over the API; "
        f"pass its models explicitly."
    )
