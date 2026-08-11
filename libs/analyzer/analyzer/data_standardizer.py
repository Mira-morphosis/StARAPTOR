import datetime
import os
from typing import Tuple, Any, Optional

from dotenv import load_dotenv

from storage.storage_utils import db_connection, get_daily_kpis_map

load_dotenv()

MIN_REVIEWS_THRESHOLD = int(os.getenv("STARAPTOR_MINING_THRESHOLD", 500))
CHECK_DAYS_DELTA = int(os.getenv("STARAPTOR_MINING_MIN_DAYS_LOOKBACK_DELTA", 7))
MAX_LOOKBACK_DAYS = min(int(os.getenv("STARAPTOR_MAX_LOOKBACK_DAYS", 140)), 140)


class DataStandardizer:
    """Converts raw stored reviews into the transaction format used by the association-rule miner."""

    def __init__(self, app_id: int):
        self.app_id = app_id
        self.kpi_map = get_daily_kpis_map(app_id)

    @property
    def sliding_window_size(self) -> Tuple[int, int]:
        """
        Finds the smallest lookback window (in days) that contains at least MIN_REVIEWS_THRESHOLD reviews.
        :return: A tuple (window_days, cutoff_timestamp) describing the chosen window.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        with db_connection(self.app_id) as conn:
            for d in range(CHECK_DAYS_DELTA, MAX_LOOKBACK_DAYS + 1, CHECK_DAYS_DELTA):
                dt_limit = int((now - datetime.timedelta(days=d)).timestamp())

                row = conn.execute(
                    "SELECT COUNT(*) FROM reviews WHERE timestamp_updated >= ?",
                    [dt_limit]
                ).fetchone()

                count = row[0] if row is not None else 0

                if count >= MIN_REVIEWS_THRESHOLD:
                    return d, dt_limit

            max_dt_limit = int((now - datetime.timedelta(days=MAX_LOOKBACK_DAYS)).timestamp())
            return MAX_LOOKBACK_DAYS, max_dt_limit

    def get_historical_sliding_windows(
            self,
            window_days: int = 14,
            step_days: int = 7
    ) -> list[Tuple[list[list[str]], dict[str, Any]]]:
        """
        Builds a series of overlapping historical windows of transactions for backfilling mining runs.
        :param window_days: The size in days of each window.
        :param step_days: The number of days to shift back between consecutive windows.
        :return: A list of (transactions, metadata) tuples, one per window that met MIN_REVIEWS_THRESHOLD.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        max_limit_ts = int((now - datetime.timedelta(days=MAX_LOOKBACK_DAYS)).timestamp())

        datasets = []
        current_end_dt = now

        with db_connection(self.app_id) as conn:
            while True:
                current_start_dt = current_end_dt - datetime.timedelta(days=window_days)
                start_ts = int(current_start_dt.timestamp())
                end_ts = int(current_end_dt.timestamp())

                if start_ts < max_limit_ts:
                    break

                cursor = conn.execute("""
                                      SELECT *
                                      FROM reviews
                                      WHERE timestamp_updated >= ?
                                        AND timestamp_updated < ?
                                        AND isSpam = FALSE
                                      ORDER BY timestamp_updated DESC
                                      """, [start_ts, end_ts])

                columns = [desc[0] for desc in cursor.description]
                reviews = [dict(zip(columns, row)) for row in cursor.fetchall()]

                if len(reviews) >= MIN_REVIEWS_THRESHOLD:
                    transactions = [self.transactional(r) for r in reviews]
                    metadata = {
                        "app_id": self.app_id,
                        "window_size_days": window_days,
                        "total_reviews": len(reviews),
                        "timestamp_start": start_ts,
                        "timestamp_end": end_ts,
                        "window_label": f"{current_start_dt.strftime('%Y%m%d')}_{current_end_dt.strftime('%Y%m%d')}"
                    }
                    datasets.append((transactions, metadata))

                current_end_dt -= datetime.timedelta(days=step_days)

        return datasets

    def get_window_review_list(self, begin_timestamp: int) -> list[dict[str, Any]]:
        """
        Fetches all non-spam reviews updated on or after the given timestamp.
        :param begin_timestamp: The Unix timestamp lower bound.
        :return: A list of review records as dicts.
        """
        with db_connection(self.app_id) as conn:
            cursor = conn.execute("""
                                  SELECT *
                                  FROM reviews
                                  WHERE timestamp_updated >= ?
                                    AND isSpam = FALSE
                                  ORDER BY timestamp_updated DESC
                                  """, [begin_timestamp])
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    @staticmethod
    def discretize_playtime(minutes: int) -> str:
        """
        Buckets a raw playtime-at-review value into a coarse playtime category.
        :param minutes: Playtime in minutes at the time the review was written.
        :return: A category label string.
        """
        if minutes < 120:
            return "playtime_refund_zone"
        elif minutes < 600:
            return "playtime_casual"
        elif minutes < 3000:
            return "playtime_mid_core"
        elif minutes < 12000:
            return "playtime_heavy"
        else:
            return "playtime_hardcore"

    def transactional(self, review: dict[str, Any]) -> list[str]:
        """
        Converts a single review record into a "transaction": a list of categorical item tags.

        Armchair rundown: association-rule mining (the FP-Growth step later in the pipeline)
        works on "shopping basket" data — each transaction is a set of items, and the miner
        looks for items that tend to appear together. Here, each review is a basket, and its
        items are things like "gameplay_4", "voted_up_True", or "playtime_casual". This
        function is what turns a review row into that basket.
        :param review: A single review record as a dict.
        :return: A list of string tags representing this review's basket items.
        """
        transaction = []

        pillars = [
            "multiplayer", "immersion", "community", "replayability", "story",
            "monetizationModel", "gameplay", "controls", "graphics", "customization",
            "audio", "difficulty"
        ]
        for pillar in pillars:
            v = review.get(pillar)
            if v is not None:
                transaction.append(f"{pillar}_{v}")

        booleans = [
            "steam_purchase", "received_for_free", "voted_up", "primarily_steam_deck",
            "isUseful", "containsPositiveAspects", "containsNegativeAspects",
            "actuallyRecommendsTitle", "containsBugDescription"
        ]
        for field in booleans:
            v = review.get(field)
            if v is not None:
                transaction.append(f"{field}_{v}")

        playtime = review.get("playtime_at_review")
        if playtime is not None:
            transaction.append(self.discretize_playtime(int(str(playtime))))

        lang = review.get("language")
        if lang:
            transaction.append(f"lang_{lang}")

        if review.get("has_hardware_info"):
            for hw_col in ["hw_ram", "hw_vram", "hw_os"]:
                val: Optional[str] = review.get(hw_col)
                if isinstance(val, str) and val not in ("N/A", "Unknown"):
                    transaction.append(f"{hw_col}_{val.replace(' ', '_')}")

        ts = review.get("timestamp_updated")
        if ts is not None:
            review_date = datetime.datetime.fromtimestamp(float(str(ts)), tz=datetime.timezone.utc).date()
            trend_tag = self.kpi_map.get(review_date)
            if trend_tag:
                transaction.append(f"kpi_{trend_tag}")

        return transaction

    def get_mining_dataset(self) -> Tuple[list[list[str]], dict[str, Any]]:
        """
        Builds the transaction list and metadata for the current best sliding window.
        :return: A tuple (transactions, metadata) ready to be passed into the FP-Growth miner.
        """
        reviews = self.get_window_review_list(self.sliding_window_size[1])
        transactions = [self.transactional(r) for r in reviews]

        metadata = {
            "app_id": self.app_id,
            "window_size_days": self.sliding_window_size[0],
            "total_reviews": len(reviews),
            "timestamp_start": self.sliding_window_size[1]
        }

        return transactions, metadata