import instructor
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()
provider_type = os.getenv("STARAPTOR_PROVIDER_TYPE", "openai_compatible").lower()
base_url = os.getenv("STARAPTOR_BASE_URL") or None
api_key = os.getenv("STARAPTOR_API_KEY", "staraptor")
mode_str = os.getenv("STARAPTOR_INSTRUCTOR_MODE", "JSON")


v1_url = base_url+"v1" if base_url is not None else None

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