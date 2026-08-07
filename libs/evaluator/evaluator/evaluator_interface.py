import asyncio
import os
from typing import List, Dict, Any, Optional

import requests
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio

from evaluator.instructor_factory import get_instructor_client, get_available_models, base_url


class ValidReviewAnalysis(BaseModel):
    isSpam: bool = Field(
        ...,
        description="Set to True ONLY if the review is spam, ASCII art, recipes, repetitive memes/copypasta, or joke texts lacking real feedback. Otherwise False."
    )
    multiplayer: Optional[int] = Field(default=None, ge=1, le=5)
    immersion: Optional[int] = Field(default=None, ge=1, le=5)
    community: Optional[int] = Field(default=None, ge=1, le=5)
    replayability: Optional[int] = Field(default=None, ge=1, le=5)
    story: Optional[int] = Field(default=None, ge=1, le=5)
    monetizationModel: Optional[int] = Field(default=None, ge=1, le=5)
    gameplay: Optional[int] = Field(default=None, ge=1, le=5)
    controls: Optional[int] = Field(default=None, ge=1, le=5)
    graphics: Optional[int] = Field(default=None, ge=1, le=5)
    customization: Optional[int] = Field(default=None, ge=1, le=5)
    audio: Optional[int] = Field(default=None, ge=1, le=5)
    difficulty: Optional[int] = Field(default=None, ge=1, le=5)
    isUseful: Optional[bool] = Field(default=None)
    containsPositiveAspects: Optional[bool] = Field(default=None)
    containsNegativeAspects: Optional[bool] = Field(default=None)
    actuallyRecommendsTitle: Optional[bool] = Field(default=None)
    containsBugDescription: Optional[bool] = Field(default=None)


ReviewResponse = ValidReviewAnalysis

SYSTEM_PROMPT = (
    "You are an objective, un-biased Steam review parser. Your job is to strictly populate the appropriate schema.\n"
    "CRITICAL ROUTING & SCORING RULES:\n"
    "1. If spam, set 'isSpam': true and OMIT all other keys from the JSON.\n"
    "2. DO NOT BE LENIENT. For valid reviews, LLMs suffer from positivity bias; correct this by actively using 1 and 2 for poor or problematic experiences.\n"
        "For scores (1-5), ONLY include keys that are explicitly discussed in the review. Omit any key that is not mentioned (do not output null)."
    "3. Numeric Scales (1-5):\n"
    "   - 5 = Flawless / Excellent\n"
    "   - 4 = Good / Positive\n"
    "   - 3 = Average / Mixed\n"
    "   - 2 = Bad / Frustrating / Significantly flawed\n"
    "   - 1 = Terrible / Completely broken / Unplayable / Hostile\n"
    "4. Only extract a score if the aspect is explicitly discussed or directly inferable. Otherwise, leave it null.\n"
    "5. Sarcasm / Irony: Decode the underlying intent. If a user mockingly says 'I love getting killed by cheaters 10/10', score the aspect (e.g., community/multiplayer) as a 1.\n"
    "6. 'actuallyRecommendsTitle' must strictly match the review text's sentiment, ignoring Steam's official recommendation status."
)
def count_prompt_tokens() -> int:
    response = requests.post(
        url = base_url+"tokenize",
        json = {"content": SYSTEM_PROMPT}
    )

    tokens = response.json().get("tokens", [])
    return len(tokens)

class ReviewProcessor:
    def __init__(self):
        self.client = get_instructor_client()
        concurrency_limit = int(os.getenv("STARAPTOR_CONCURRENCY_LIMIT", "100"))
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def _analyze_single(self, model: str, review: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        review_text = review.get("review", "").strip()
        if not review_text:
            return None

        async with self.semaphore:
            try:
                # Use the Union type here
                result: ReviewResponse = await self.client.chat.completions.create(
                    model=model,
                    response_model=ReviewResponse,  # Instructor handles the choice dynamically
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Review text to analyze:\n{review_text}"}
                    ],
                    temperature=0.0,
                    max_retries=1,
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": False
                        },
                        "reasoning_budget": 0
                    }
                )

                res_dict = result.model_dump()

                # If it matched the SpamReview schema, discard immediately
                if res_dict.get("isSpam") is True:
                    return None

                return {**review, **res_dict}

            except Exception:
                return None

    async def process_batch(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not reviews:
            return []

        available_models = await get_available_models()
        if not available_models:
            raise RuntimeError("No models available from the configured provider.")

        target_model = available_models[0]

        tasks = [self._analyze_single(target_model, r) for r in reviews]
        raw_results = await tqdm_asyncio.gather(
            *tasks,
            desc="Processing Reviews",
            unit="review"
        )

        return [item for item in raw_results if item is not None]