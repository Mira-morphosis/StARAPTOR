import os
import asyncio
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from evaluator.instructor_factory import get_instructor_client, get_available_models

class SpamReview(BaseModel):
    isSpam: Literal[True] = Field(
        True,
        description="Set to True ONLY if the review is spam, ASCII art, recipes, repetitive memes/copypasta, or joke texts lacking real feedback."
    )

class ValidReviewAnalysis(BaseModel):
    isSpam: Literal[False] = Field(False, description="Must be False for genuine reviews with meaningful text.")
    multiplayer: Optional[int] = Field(None, ge=1, le=5)
    immersion: Optional[int] = Field(None, ge=1, le=5)
    community: Optional[int] = Field(None, ge=1, le=5)
    replayability: Optional[int] = Field(None, ge=1, le=5)
    story: Optional[int] = Field(None, ge=1, le=5)
    monetizationModel: Optional[int] = Field(None, ge=1, le=5)
    gameplay: Optional[int] = Field(None, ge=1, le=5)
    controls: Optional[int] = Field(None, ge=1, le=5)
    graphics: Optional[int] = Field(None, ge=1, le=5)
    customization: Optional[int] = Field(None, ge=1, le=5)
    audio: Optional[int] = Field(None, ge=1, le=5)
    difficulty: Optional[int] = Field(None, ge=1, le=5)
    isUseful: bool
    containsPositiveAspects: bool
    containsNegativeAspects: bool
    actuallyRecommendsTitle: bool
    containsBugDescription: bool


ReviewResponse = Union[SpamReview, ValidReviewAnalysis]

SYSTEM_PROMPT = (
    "You are an objective, un-biased Steam review parser. Your job is to strictly populate the appropriate schema.\n"
    "CRITICAL ROUTING & SCORING RULES:\n"
    "1. SPAM FILTER: If the review is ASCII art, a recipe, repetitive memes/copypasta, or a joke text lacking real feedback, use the SpamReview schema and immediately stop.\n"
    "2. DO NOT BE LENIENT. For valid reviews, LLMs suffer from positivity bias; correct this by actively using 1 and 2 for poor or problematic experiences.\n"
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
        raw_results = await asyncio.gather(*tasks)

        return [item for item in raw_results if item is not None]