#!/usr/bin/env python3
import os
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timedelta

import pandas as pd
import httpx
from fastapi import FastAPI, Request, Query, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Internal StARAPTOR modules
from orchestrator.orchestrator_service import (
    orchestrator_service,
    OperationLockedError,
    FORECAST_DAYS,
)
from storage.storage_utils import (
    db_connection,
    get_latest_run_id,
    list_mining_runs,
    DB_ROOT
)
from dashboard.rule_translator import (
    translate_rule_to_natural_language,
    _split_items,
    LANG_NAMES,
)

app = FastAPI(title="StARAPTOR Dashboard API", version="1.0.0")

PORT = int(os.getenv("STARAPTOR_DASHBOARD_PORT", 8001))
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

DEFAULT_APP_ID = int(os.getenv("STARAPTOR_APP_ID", 1086940))
logger = logging.getLogger("staraptor.dashboard")

STEAM_GAMES_CACHE: Dict[int, Dict[str, str]] = {}
DAILY_UPDATES_ENABLED: Dict[int, bool] = {}
TODAY_SKIPPED: Dict[int, str] = {}
IS_MANUAL_RUN: Dict[int, bool] = {}

# --- Review list display -----------------------------------------------------
# Page size for the reviews page. The list is paginated server-side so it stays
# fast no matter how many years of reviews accumulate in the archive.
REVIEWS_PAGE_SIZE = 25


def _latest_forecast_path(app_id: int) -> Path:
    return Path(DB_ROOT).expanduser() / "exports" / f"{app_id}_forecast_latest.csv"


def _normalize_forecast_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["day", "predicted_mean"])

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "day" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "day"})
        elif "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "day"})

    if "predicted_mean" not in df.columns:
        if "predicted_mean_concurrent" in df.columns:
            df = df.rename(columns={"predicted_mean_concurrent": "predicted_mean"})
        elif "kpi" in df.columns:
            df = df.rename(columns={"kpi": "predicted_mean"})

    if "day" not in df.columns or "predicted_mean" not in df.columns:
        return pd.DataFrame(columns=["day", "predicted_mean"])

    df = df[["day", "predicted_mean"]].copy()
    df["day"] = pd.to_datetime(df["day"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["predicted_mean"] = pd.to_numeric(df["predicted_mean"], errors="coerce")
    return df.dropna(subset=["day", "predicted_mean"])


async def _safe_export_forecast(app_id: int, days: int):
    try:
        await orchestrator_service.export_forecast(app_id=app_id, days=days)
    except OperationLockedError:
        logger.warning("Forecast export skipped because another operation is running.")
    except Exception:
        logger.exception("Forecast export failed.")


def needs_initialization(app_id: int) -> bool:
    """Verifies whether the App ID requires complete cold-start initialization."""
    db_path = Path(DB_ROOT).expanduser() / f"{app_id}.db"
    if not db_path.exists():
        return True
    try:
        runs = list_mining_runs(app_id)
        return runs.empty
    except Exception:
        return True


async def run_manual_pipeline_task(app_id: int):
    """Wrapper to run the manual pipeline and restore state upon completion."""
    IS_MANUAL_RUN[app_id] = True
    try:
        if needs_initialization(app_id):
            await orchestrator_service.run_initial_pipeline(app_id=app_id, verbose=False)
        else:
            await orchestrator_service.run_daily_pipeline(app_id=app_id, verbose=False)
    finally:
        IS_MANUAL_RUN[app_id] = False


async def get_steam_game_info(app_id: int) -> Dict[str, str]:
    if app_id in STEAM_GAMES_CACHE:
        return STEAM_GAMES_CACHE[app_id]

    name = f"App ID {app_id}"
    icon_url = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{app_id}/header.jpg"

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True, headers=headers) as client:
            api_url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
            resp = await client.get(api_url)
            if resp.status_code == 200:
                data = resp.json()
                app_data = data.get(str(app_id), {})
                if app_data.get("success") and "data" in app_data:
                    info_data = app_data["data"]
                    name = info_data.get("name", name)
                    icon_url = info_data.get("header_image") or info_data.get("capsule_image") or icon_url
    except Exception as e:
        logger.warning(f"Unable to fetch game name from Steam for App ID {app_id}: {e}")

    info = {"name": name, "icon_url": icon_url}
    STEAM_GAMES_CACHE[app_id] = info
    return info


def should_auto_start_update(app_id: int) -> bool:
    if not DAILY_UPDATES_ENABLED.get(app_id, True):
        return False

    today_str = datetime.now().strftime("%Y-%m-%d")
    if TODAY_SKIPPED.get(app_id) == today_str:
        return False

    runs = list_mining_runs(app_id)
    if runs.empty or "timestamp" not in runs.columns:
        return True

    last_dt = runs["timestamp"].max()
    if pd.isnull(last_dt):
        return True

    if hasattr(last_dt, "to_pydatetime"):
        last_dt = last_dt.to_pydatetime()

    if last_dt.tzinfo is not None:
        last_dt = last_dt.replace(tzinfo=None)

    return (datetime.now() - last_dt) > timedelta(hours=24)


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request, app_id: Optional[int] = Query(None)):
    selected_app_id = app_id or DEFAULT_APP_ID
    return templates.TemplateResponse(request, "index.html", {"app_id": selected_app_id})


@app.get("/api/banner-status", response_class=HTMLResponse)
async def get_banner_status(background_tasks: BackgroundTasks, app_id: int = DEFAULT_APP_ID):
    is_enabled = DAILY_UPDATES_ENABLED.get(app_id, True)
    is_manual = IS_MANUAL_RUN.get(app_id, False)

    if is_enabled and not orchestrator_service.is_running:
        if needs_initialization(app_id):
            background_tasks.add_task(orchestrator_service.run_initial_pipeline, app_id=app_id, verbose=False)
        elif should_auto_start_update(app_id):
            background_tasks.add_task(orchestrator_service.run_daily_pipeline, app_id=app_id, verbose=False)

    if orchestrator_service.is_running:
        progress = getattr(orchestrator_service, "progress", 0)
        title_text = "Manual update is running:" if is_manual else "Scheduled daily update is running:"

        disable_button_html = "" if is_manual else f"""
<button
    type="button"
    class="text-white/80 hover:text-white underline text-xs sm:text-sm font-medium transition cursor-pointer shrink-0"
    hx-post="/api/toggle-daily-updates?app_id={app_id}"
    hx-target="#scheduled-update-banner"
    hx-swap="outerHTML">
    Disable daily updates
</button>
"""

        banner_html = f"""
<div id="scheduled-update-banner"
     hx-get="/api/banner-status?app_id={app_id}"
     hx-trigger="every 2s, updateBanner from:body"
     hx-swap="outerHTML"
     class="flex flex-col sm:flex-row items-center justify-between px-4 sm:px-6 py-3 sm:py-0 w-full min-h-[60px] sm:h-[64px] bg-gradient-to-t from-[#2c4a61] to-[#396584] shadow-[0_4px_30px_rgba(0,0,0,0.7)] border-b border-white/10 gap-3">

    <div class="flex flex-wrap items-center gap-3 sm:gap-4 w-full sm:w-auto">
        <span class="text-white text-sm sm:text-base font-medium leading-tight shrink-0">
            {title_text}
        </span>

        <div class="relative w-full max-w-[280px] sm:w-[320px] h-[10px] bg-[#171d25] overflow-hidden border border-white/10 shadow-inner">
            <div class="h-full bg-gradient-to-r from-[#3389e4] via-[#47bfff] to-[#3389e4] animate-shift-gradient transition-all duration-300"
                 style="width: {progress}%;"></div>
        </div>

        <button
            type="button"
            class="text-white/80 hover:text-red-400 transition cursor-pointer shrink-0 flex items-center justify-center p-0.5"
            hx-post="/api/skip-today-update?app_id={app_id}"
            hx-target="#scheduled-update-banner"
            hx-swap="outerHTML"
            title="Stop update">
            <svg class="size-6 fill-current" viewBox="0 0 24 24">
                <path d="M6 6h12v12H6z"/>
            </svg>
        </button>
    </div>

    {disable_button_html}
</div>
"""
        return HTMLResponse(content=banner_html)

    empty_container = f"""
<div id="scheduled-update-banner"
     hx-get="/api/banner-status?app_id={app_id}"
     hx-trigger="every 5s, updateBanner from:body"
     hx-swap="outerHTML"></div>
"""
    return HTMLResponse(content=empty_container)


@app.post("/api/skip-today-update", response_class=HTMLResponse)
async def skip_today_update(app_id: int = DEFAULT_APP_ID):
    today_str = datetime.now().strftime("%Y-%m-%d")
    TODAY_SKIPPED[app_id] = today_str

    if hasattr(orchestrator_service, "cancel"):
        orchestrator_service.cancel()
    elif hasattr(orchestrator_service, "stop"):
        orchestrator_service.stop()

    empty_container = f"""
<div id="scheduled-update-banner"
     hx-get="/api/banner-status?app_id={app_id}"
     hx-trigger="every 5s, updateBanner from:body"
     hx-swap="outerHTML"></div>
"""
    return HTMLResponse(content=empty_container)


@app.post("/api/toggle-daily-updates", response_class=HTMLResponse)
async def toggle_daily_updates(app_id: int = DEFAULT_APP_ID):
    current_status = DAILY_UPDATES_ENABLED.get(app_id, True)
    DAILY_UPDATES_ENABLED[app_id] = not current_status

    if not DAILY_UPDATES_ENABLED[app_id]:
        if hasattr(orchestrator_service, "cancel"):
            orchestrator_service.cancel()
        elif hasattr(orchestrator_service, "stop"):
            orchestrator_service.stop()

    empty_container = f"""
<div id="scheduled-update-banner"
     hx-get="/api/banner-status?app_id={app_id}"
     hx-trigger="every 5s, updateBanner from:body"
     hx-swap="outerHTML"></div>
"""
    return HTMLResponse(content=empty_container)


@app.get("/api/current-game", response_class=HTMLResponse)
async def get_current_game(request: Request, app_id: int = DEFAULT_APP_ID):
    db_folder = Path(DB_ROOT).expanduser()
    available_games = []

    if db_folder.exists() and db_folder.is_dir():
        for db_file in db_folder.glob("*.db"):
            try:
                aid = int(db_file.stem)
                info = await get_steam_game_info(aid)
                available_games.append({
                    "name": info["name"],
                    "db_filename": db_file.name,
                    "is_active": (aid == app_id),
                    "app_id": aid
                })
            except ValueError:
                continue

    current_info = await get_steam_game_info(app_id)

    if not any(g["app_id"] == app_id for g in available_games):
        available_games.append({
            "name": current_info["name"],
            "db_filename": f"{app_id}.db",
            "is_active": True,
            "app_id": app_id
        })

    current_game = {
        "name": current_info["name"],
        "icon_url": current_info["icon_url"],
        "app_id": app_id
    }

    return templates.TemplateResponse(
        request,
        "components/game_selector.jinja2",
        {
            "current_game": current_game,
            "available_games": available_games
        }
    )


@app.get("/api/last-updated", response_class=HTMLResponse)
async def get_last_updated(app_id: int = DEFAULT_APP_ID):
    if orchestrator_service.is_running:
        status_text = "UPDATE IN PROGRESS..."
    else:
        runs = list_mining_runs(app_id)
        if not runs.empty and "timestamp" in runs.columns:
            last_dt = runs["timestamp"].max()
            status_text = last_dt.strftime("%Y-%m-%d %H:%M") if pd.notnull(last_dt) else "N/A"
        else:
            status_text = "N/A"

    html_content = f"""
<span class="text-white font-semibold text-[8px] sm:text-[9px] leading-[1.1]">LAST UPDATED:</span>
<span class="text-white/70 text-[11px] sm:text-xs leading-[1.1]">{status_text}</span>
"""
    return HTMLResponse(content=html_content)


@app.get("/api/associations", response_class=HTMLResponse)
async def get_associations(request: Request, app_id: int = DEFAULT_APP_ID):
    latest_run = get_latest_run_id(app_id)
    associations = []

    if latest_run:
        with db_connection(app_id) as conn:
            query = """
                SELECT antecedents, consequents, lift, confidence
                FROM association_rules
                WHERE run_id = ?
                ORDER BY lift DESC LIMIT 15
            """
            rows = conn.execute(query, [latest_run]).fetchall()

            for row in rows:
                ant, cons, lift, conf = row
                natural_text = translate_rule_to_natural_language(ant, cons, lift, conf)

                if lift >= 1.5:
                    category = "critical"
                elif lift >= 1.2:
                    category = "warning"
                elif lift >= 1.0:
                    category = "positive"
                else:
                    category = "neutral"

                associations.append({
                    "text": natural_text,
                    "category": category,
                    # Raw itemsets are passed through untouched so the review
                    # modal can filter on the rule's EXACT attributes. The tags
                    # never contain commas or spaces, so they survive URL encoding.
                    "antecedents": ant,
                    "consequents": cons,
                })

    return templates.TemplateResponse(
        request,
        "components/associations_list.jinja2",
        {
            "associations": associations,
            "app_id": app_id,
        }
    )

@app.get("/data/forecast_data.csv")
async def get_forecast_csv(
        app_id: int = DEFAULT_APP_ID,
        days: int = FORECAST_DAYS,
        refresh: bool = False
):
    latest_path = _latest_forecast_path(app_id)

    if refresh or not latest_path.exists():
        if not orchestrator_service.is_running:
            await _safe_export_forecast(app_id, days)

    if latest_path.exists():
        try:
            df = pd.read_csv(latest_path)
            df = _normalize_forecast_df(df)

            if not df.empty:
                return Response(
                    content=df.to_csv(index=False),
                    media_type="text/csv"
                )
        except Exception:
            logger.exception("Unable to read forecast export.")

    return Response(
        content="day,predicted_mean\n",
        media_type="text/csv"
    )


@app.post("/api/export-forecast", response_class=HTMLResponse)
async def export_forecast_endpoint(
        request: Request,
        response: Response,
        background_tasks: BackgroundTasks,
        app_id: int = DEFAULT_APP_ID,
        days: int = FORECAST_DAYS
):
    if not orchestrator_service.is_running:
        background_tasks.add_task(_safe_export_forecast, app_id, days)

    response.headers["HX-Trigger"] = "updateLastUpdated, updateBanner"
    return await get_associations(request, app_id)


# --- Rule-tag -> SQL translation -------------------------------------------
# A mined rule is a set of categorical tags (see DataStandardizer.transactional).
# To show "the reviews that make the rule true" we translate every tag into a
# predicate on the reviews table and AND them all together. The column sets below
# must stay in sync with data_standardizer.DataStandardizer.transactional.

_RULE_PILLAR_COLS = [
    "multiplayer", "immersion", "community", "replayability", "story",
    "monetizationModel", "gameplay", "controls", "graphics", "customization",
    "audio", "difficulty",
]
_RULE_BOOL_COLS = [
    "steam_purchase", "received_for_free", "voted_up", "primarily_steam_deck",
    "isUseful", "containsPositiveAspects", "containsNegativeAspects",
    "actuallyRecommendsTitle", "containsBugDescription",
]
_RULE_HW_COLS = ["hw_os", "hw_ram", "hw_vram", "hw_gpu", "hw_cpu_vendor", "hw_resolution"]

# Playtime buckets mirror DataStandardizer.discretize_playtime (minutes at review).
_RULE_PLAYTIME_RANGES = {
    "refund_zone": "playtime_at_review < 120",
    "casual": "playtime_at_review >= 120 AND playtime_at_review < 600",
    "mid_core": "playtime_at_review >= 600 AND playtime_at_review < 3000",
    "heavy": "playtime_at_review >= 3000 AND playtime_at_review < 12000",
    "hardcore": "playtime_at_review >= 12000",
}


def _tag_to_sql(tag: str):
    """Translate one rule tag into a SQL predicate.

    :param tag: A single categorical tag, e.g. ``"gameplay_4"``, ``"voted_up_True"``,
        ``"hw_os_Windows_11"``, ``"lang_russian"``, ``"playtime_heavy"`` or
        ``"kpi_retention_growth"``.
    :return: A ``(sql_fragment, params)`` tuple, or ``None`` if the tag is unknown
        (an unknown tag is skipped rather than producing a broken query).
    """
    t = tag.strip()
    low = t.lower()

    if low.startswith("playtime_"):
        bucket = low[len("playtime_"):]
        frag = _RULE_PLAYTIME_RANGES.get(bucket)
        if not frag:
            return None  # unknown playtime bucket -> no-op
        return f"({frag})", []

    if low.startswith("lang_"):
        return "language = ?", [t[len("lang_"):]]

    if low.startswith("kpi_"):
        # The KPI trend tag encodes the daily_kpis.trend of the review's UTC date.
        frag = ("EXISTS (SELECT 1 FROM daily_kpis d WHERE d.date = "
                "(to_timestamp(timestamp_updated) AT TIME ZONE 'UTC')::DATE "
                "AND d.trend = ?)")
        return frag, [low[len("kpi_"):]]

    # Hardware: longest known column prefix wins (hw_os before a hypothetical hw_o).
    for col in sorted(_RULE_HW_COLS, key=len, reverse=True):
        if low.startswith(col + "_"):
            value = t[len(col) + 1:].replace("_", " ")
            return f"{col} = ?", [value]

    # Booleans: tag is "<column>_True|False".
    for col in sorted(_RULE_BOOL_COLS, key=len, reverse=True):
        prefix = col + "_"
        if t.startswith(prefix):
            value = t[len(prefix):]
            if value in ("True", "False"):
                return f"{col} = ?", [value == "True"]

    # Score pillars: tag is "<column>_<int>".
    for col in sorted(_RULE_PILLAR_COLS, key=len, reverse=True):
        prefix = col + "_"
        if t.startswith(prefix):
            value = t[len(prefix):]
            if value.isdigit():
                return f"{col} = ?", [int(value)]

    logger.warning("Unrecognized rule tag %r; skipping it in the review filter.", tag)
    return None


def build_reviews_filter(antecedents: Optional[str], consequents: Optional[str]):
    """Translate a rule's itemset into a WHERE clause for the reviews table.

    A rule is "true" for a review when the review carries every attribute on
    BOTH sides of the rule (the full itemset). Each tag becomes an AND-ed
    predicate, so the result is exactly the set of matching reviews. With no
    tags (e.g. the unfiltered "Read Reviews" entry point) an empty clause is
    returned and all reviews are listed.

    :param antecedents: The rule's antecedent itemset string (may be empty).
    :param consequents: The rule's consequent itemset string (may be empty).
    :return: A ``(where_clause, params)`` tuple; the clause includes the leading
        ``WHERE`` keyword when non-empty.
    """
    tags = _split_items(antecedents or "") + _split_items(consequents or "")

    conditions, params = [], []
    for tag in tags:
        result = _tag_to_sql(tag)
        if result:
            frag, p = result
            conditions.append(frag)
            params.extend(p)

    where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    return where_clause, params


_REVIEW_COLUMNS = (
    "recommendationid, review, language, voted_up, weighted_vote_score, "
    "playtime_at_review, multiplayer, immersion, community, replayability, "
    "story, monetizationModel, gameplay, controls, graphics, customization, "
    "audio, difficulty, steam_purchase, received_for_free, primarily_steam_deck, "
    "isUseful, containsPositiveAspects, containsNegativeAspects, "
    "actuallyRecommendsTitle, containsBugDescription, timestamp_updated"
)


def _format_review_row(r: tuple) -> dict:
    """Map one reviews-table row onto the fields the review card renders."""
    (rid, text, language, voted_up, score, playtime, multiplayer, immersion,
     community, replayability, story, monetization_model, gameplay, controls,
     graphics, customization, audio, difficulty, steam_purchase, received_for_free,
     steam_deck, is_useful, has_positive, has_negative, recommends, reports_bugs,
     ts) = r

    # "Published 2 Sept." (Steam-style); fall back gracefully for odd timestamps.
    date_str = ""
    if ts:
        try:
            dt = datetime.fromtimestamp(ts)
            date_str = f"Published {dt.strftime('%-d %b')}"
        except Exception:
            pass

    pillars = [
        ("Multiplayer", multiplayer),
        ("Immersion", immersion),
        ("Community", community),
        ("Replayability", replayability),
        ("Story", story),
        ("Monetization", monetization_model),
        ("Gameplay", gameplay),
        ("Controls", controls),
        ("Graphics", graphics),
        ("Customization", customization),
        ("Audio", audio),
        ("Difficulty", difficulty),
    ]

    return {
        "id": rid,
        "text": text or "",
        "language": LANG_NAMES.get((language or "").lower(), (language or "").title()),
        "voted_up": bool(voted_up),
        "score": round(score or 0.0, 2),
        "playtime_hours": round((playtime or 0) / 60, 1),
        "steam_purchase": bool(steam_purchase),
        "received_for_free": bool(received_for_free),
        "steam_deck": bool(steam_deck),
        "is_useful": bool(is_useful),
        "has_positive": bool(has_positive),
        "has_negative": bool(has_negative),
        "recommends": bool(recommends),
        "reports_bugs": bool(reports_bugs),
        "pillars": pillars,
        "date_str": date_str,
        # Whether any pillar rating is present (drives the ratings strip).
        "has_pillars": any(v is not None for _, v in pillars),
    }


def fetch_reviews_page(
    app_id: int,
    where_clause: str,
    params: list,
    page: int,
    page_size: int = REVIEWS_PAGE_SIZE,
):
    """Fetch one page of reviews matching a filter, newest first.

    :return: A ``(rows, total_count)`` tuple where *rows* is a list of dicts
        shaped by ``_format_review_row`` and *total_count* is the number of
        reviews matching the whole filter (used for pagination).
    """
    offset = max(page - 1, 0) * page_size
    with db_connection(app_id) as conn:
        total = conn.execute(
            f"SELECT COUNT(*) FROM reviews{where_clause}", params
        ).fetchone()[0]

        query = (
            f"SELECT {_REVIEW_COLUMNS} FROM reviews{where_clause} "
            "ORDER BY timestamp_updated DESC, recommendationid DESC "
            f"LIMIT {page_size} OFFSET {offset}"
        )
        rows = [_format_review_row(r) for r in conn.execute(query, params).fetchall()]

    return rows, int(total or 0)


@app.get("/reviews", response_class=HTMLResponse)
async def get_reviews_page(
        request: Request,
        app_id: int = DEFAULT_APP_ID,
        antecedents: Optional[str] = Query(None),
        consequents: Optional[str] = Query(None),
        category: Optional[str] = Query(None),
        page: int = Query(1, ge=1)
):
    """The Reviews page.

    Without ``antecedents``/``consequents`` it lists all of the game's reviews;
    with them (as linked from an association bar) it lists only the reviews
    that make that rule true. The filter is carried in the URL, so closing the
    banner (which links back to the unfiltered page) simply clears it.
    """
    game_info = await get_steam_game_info(app_id)

    # Translate the rule's itemset into a WHERE clause. A rule is "true" for a
    # review when the review carries every attribute on BOTH sides of the rule,
    # so we filter on the full itemset (antecedents + consequents).
    where_clause, params = build_reviews_filter(antecedents, consequents)

    try:
        reviews_data, total_count = fetch_reviews_page(app_id, where_clause, params, page)
    except Exception:
        logger.exception("Failed to fetch reviews for app %s.", app_id)
        reviews_data, total_count = [], 0

    # The banner is only shown when a filter is actually applied; otherwise the
    # page simply lists all reviews.
    has_filter = bool(antecedents or consequents)
    rule_display = (
        translate_rule_to_natural_language(antecedents or "", consequents or "")
        if has_filter else ""
    )

    total_pages = max(1, -(-total_count // REVIEWS_PAGE_SIZE))  # ceil division
    page = min(page, total_pages)

    return templates.TemplateResponse(
        request,
        "reviews.html",
        {
            "app_id": app_id,
            "game_name": game_info["name"],
            "icon_url": game_info["icon_url"],
            "has_filter": has_filter,
            "rule_display": rule_display,
            "antecedents": antecedents or "",
            "consequents": consequents or "",
            "category": category or "",
            "reviews_data": reviews_data,
            "total_count": total_count,
            "page": page,
            "total_pages": total_pages,
        }
    )


@app.post("/api/refresh", response_class=HTMLResponse)
async def refresh_data(
        request: Request,
        response: Response,
        background_tasks: BackgroundTasks,
        app_id: int = DEFAULT_APP_ID
):
    if not orchestrator_service.is_running:
        background_tasks.add_task(run_manual_pipeline_task, app_id)

    response.headers["HX-Trigger"] = "updateLastUpdated, updateBanner"
    return await get_associations(request, app_id)


def run():
    import uvicorn
    uvicorn.run("dashboard.main:app", host="0.0.0.0", port=PORT, reload=True)


if __name__ == "__main__":
    run()