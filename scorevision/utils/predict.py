from typing import Any
from time import monotonic
from json import loads, dumps
from random import uniform
from logging import getLogger
from typing import AsyncGenerator

from asyncio import TimeoutError, sleep, gather
from aiohttp import ClientError

from scorevision.chute_template.schemas import TVPredictInput, TVPredictOutput
from scorevision.utils.data_models import SVRunOutput, SVPredictResult
from scorevision.utils.settings import get_settings
from scorevision.utils.async_clients import get_async_client, get_semaphore
from scorevision.utils.challenges import prepare_challenge_payload
from scorevision.utils.chutes_helpers import (
    get_chute_slug_and_id,
    warmup_chute,
    validate_chute_integrity,
)

from pathlib import Path
import asyncio
import sys
sys.path.append('/work/bittensor/sn44/turbovision/sample_miner')
from miner import Miner

from .challenges import download_video_cached

logger = getLogger(__name__)

miner: Miner | None = None

async def call_miner_model_on_chutes(
    # slug: str,
    # chute_id: str,
    payload: TVPredictInput,
    frames: list[Any],
    frame_numbers: list[int],
) -> SVRunOutput:
    logger.info("Verifying chute model is valid")

    res = await predict_sv(payload=payload, frames=frames, frame_numbers=frame_numbers)
    
    return SVRunOutput(
        success=res.success,
        latency_ms=res.latency_seconds * 1000.0,
        predictions=res.predictions if res.success else None,
        error=res.error,
        model=res.model,
    )


async def predict_sv(
    payload: TVPredictInput,
    frames: list[Any] | None = None,
    frame_numbers: list[int] | None = None,
) -> SVPredictResult:
    settings = get_settings()

    t0 = monotonic()

    global miner
    if miner is None:
        miner = Miner(Path("/work/bittensor/sn44/turbovision/sample_miner"))

    logger.info(f"Calling local miner model: {len(frames)}, {len(frame_numbers)}")
    
    try:
        if frames is not None and len(frames) > 0:
            logger.info("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx111...")
            batch_images = frames
            # frame_numbers = list(range(len(batch_images)))
            logger.info(f"Using {len(batch_images)} extracted frames from payload")
        else:
            logger.info("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx222...")
            frame_numbers = list(range(settings.SCOREVISION_VIDEO_MIN_FRAME_NUMBER, settings.SCOREVISION_VIDEO_MAX_FRAME_NUMBER))
            video_name, frame_store = await download_video_cached(url=payload.url, _frame_numbers=frame_numbers)
            logger.info("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx444...")
            batch_images = []
            for fn in frame_numbers:
                frame = await asyncio.to_thread(frame_store.get_frame, fn)
                # batch_images.append(frames[fn])
                batch_images.append(frame)
            logger.info("Downloading video and extracting frames...")
                
        if not batch_images:
            raise RuntimeError("No frames extracted")
        n_keypoints = payload.meta.get("n_keypoints", 32)
        
        results = miner.predict_batch(batch_images, offset=0, n_keypoints=n_keypoints)
        
        logger.info(f"Miner produced predictions for {len(results)} frames.")
        predictions = {"frames": [r.model_dump() for r in results]}
        
        return SVPredictResult(
            success=True,
            model="local",
            latency_seconds=monotonic() - t0,
            predictions=predictions,
            error=None,
        )
    except Exception as e:
        return SVPredictResult(
            success=False,
            model=None,
            latency_seconds=monotonic() - t0,
            predictions=None,
            error=str(e),
        )


async def _warmup_from_video(
    *,
    video_url: str,
    slug: str = "demo",
    base_url: str | None = None,
):
    settings = get_settings()

    fake_chal = {
        "task_id": "warmup-fixed",
        "video_url": video_url,
        "fps": settings.SCOREVISION_VIDEO_FRAMES_PER_SECOND,
        "seed": 0,
    }

    payload, _, _, _, frame_store = await prepare_challenge_payload(challenge=fake_chal)

    async def _one():
        try:
            await predict_sv(
                payload=payload,
                slug=slug,
            )
        except Exception as e:
            logger.debug(f"warmup call error: {e}")

    await gather(*(_one() for _ in range(max(1, settings.SCOREVISION_WARMUP_CALLS))))
    frame_store.unlink()


async def warmup(url: str, slug: str) -> None:
    try:
        await _warmup_from_video(
            video_url=url,
            slug=slug,
        )
        logger.info("Warmup done.")
    except Exception as e:
        logger.error(f"Warmup errored (non-fatal): {type(e).__name__}: {e}")
