from tempfile import NamedTemporaryFile
from logging import getLogger
from pathlib import Path
from collections import OrderedDict
from threading import RLock

from contextlib import contextmanager
from cv2 import (
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_POS_FRAMES,
    COLOR_BGR2GRAY,
    COLOR_HSV2BGR,
    NORM_MINMAX,
    VideoCapture,
    calcOpticalFlowFarneback,
    cartToPolar,
    cvtColor,
    normalize,
    imwrite,
)
from numpy import ndarray, pi, zeros_like


from scorevision.utils.settings import get_settings
from scorevision.utils.async_clients import get_async_client

logger = getLogger(__name__)


@contextmanager
def open_video(path: Path) -> VideoCapture:
    logger.info(f"Attempting to open video: {path}")
    if not path.exists():
        raise FileNotFoundError
    if not path.is_file():
        raise ValueError("Path is not a file")
    video = VideoCapture(str(path))
    if not video.isOpened():
        video.release()
        raise ValueError("Could not open video")
    try:
        yield video
    finally:
        video.release()


def background_temporal_differencing(
    video_path: Path, frame_numbers: list[int]
) -> tuple[dict[int, ndarray], dict[int, ndarray]]:
    logger.info(
        f"Computing Background Temporal Differencing for frame_numbers {frame_numbers} using Dense Optical Flow..."
    )
    images, flow_images = {}, {}
    with open_video(path=video_path) as video:
        if not video.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        max_frame_number = int(video.get(CAP_PROP_FRAME_COUNT))
        prev_frame, prev_gray = None, None
        for frame_number in range(max_frame_number):
            ok, frame = video.read()
            if not ok:
                logger.error(f"Error reading frame {frame_number}")
                continue
            images[frame_number] = frame

            gray = cvtColor(frame, COLOR_BGR2GRAY)
            if frame_number in frame_numbers and prev_gray is not None:
                flow = calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                mag, ang = cartToPolar(flow[..., 0], flow[..., 1])
                hsv = zeros_like(prev_frame)
                hsv[..., 0] = ang * 180 / pi / 2
                hsv[..., 1] = 255
                hsv[..., 2] = normalize(mag, None, 0, 255, NORM_MINMAX)
                rgb = cvtColor(hsv, COLOR_HSV2BGR)

                flow_images[frame_number] = rgb

            prev_gray = gray
            prev_frame = frame

    return images, flow_images


async def download_video(
    url: str, frame_numbers: list[int]
) -> tuple[str, dict[int, ndarray], dict[int, ndarray]]:
    settings = get_settings()
    session = await get_async_client()
    async with session.get(url) as response:
        if response.status != 200:
            txt = await response.text()
            raise RuntimeError(f"Download failed {response.status}: {txt[:200]}")
        data = await response.read()

    with NamedTemporaryFile(prefix="sv_video_", suffix=".mp4") as f:
        f.write(data)

        frames, flows = background_temporal_differencing(
            video_path=Path(f.name), frame_numbers=frame_numbers
        )
    name = url.split("/")[-1]
    return name, frames, flows


class FrameStore:
    """Lazy frame/flow accessor backed by a cached MP4 on disk."""

    def __init__(
        self,
        video_path: Path,
        *,
        max_frames: int = 64,
        max_flows: int = 32,
    ) -> None:
        self.video_path = video_path
        self.video_name = video_path.name
        self._frame_cache: OrderedDict[int, ndarray] = OrderedDict()
        self._flow_cache: OrderedDict[int, ndarray] = OrderedDict()
        self._max_frames = max_frames
        self._max_flows = max_flows
        self._lock = RLock()
        self._capture: VideoCapture | None = None
        self._current_frame_index: int | None = None

    def _ensure_capture(self) -> None:
        if self._capture is None:
            cap = VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {self.video_path}")
            self._capture = cap

    def _evict_if_needed(self, cache: OrderedDict[int, ndarray], limit: int) -> None:
        if limit <= 0:
            return
        while len(cache) > limit:
            cache.popitem(last=False)

    def get_frame(self, frame_number: int) -> ndarray:
        with self._lock:
            cached = self._frame_cache.get(frame_number)
            if cached is not None:
                self._frame_cache.move_to_end(frame_number)
                return cached

            self._ensure_capture()
            if not self._capture:
                raise RuntimeError("Video capture not initialised")

            if (
                self._current_frame_index is None
                or frame_number < self._current_frame_index
            ):
                self._capture.set(CAP_PROP_POS_FRAMES, frame_number)
            elif frame_number > self._current_frame_index + 1:
                self._capture.set(CAP_PROP_POS_FRAMES, frame_number)

            ok, frame = self._capture.read()
            if not ok or frame is None:
                raise IOError(f"Failed to read frame {frame_number}")

            self._current_frame_index = frame_number
            result = frame.copy()
            self._frame_cache[frame_number] = result
            self._frame_cache.move_to_end(frame_number)
            self._evict_if_needed(self._frame_cache, self._max_frames)
            return result

    def get_flow(self, frame_number: int) -> ndarray:
        if frame_number <= 0:
            raise ValueError("Optical flow requires frame_number > 0")
        with self._lock:
            cached = self._flow_cache.get(frame_number)
            if cached is not None:
                self._flow_cache.move_to_end(frame_number)
                return cached

            prev_frame = self.get_frame(frame_number - 1)
            current_frame = self.get_frame(frame_number)

            prev_gray = cvtColor(prev_frame, COLOR_BGR2GRAY)
            gray = cvtColor(current_frame, COLOR_BGR2GRAY)
            flow = calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag, ang = cartToPolar(flow[..., 0], flow[..., 1])
            hsv = zeros_like(prev_frame)
            hsv[..., 0] = ang * 180 / pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = normalize(mag, None, 0, 255, NORM_MINMAX)
            rgb = cvtColor(hsv, COLOR_HSV2BGR)

            self._flow_cache[frame_number] = rgb
            self._flow_cache.move_to_end(frame_number)
            self._evict_if_needed(self._flow_cache, self._max_flows)
            return rgb

    def close(self) -> None:
        with self._lock:
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None
                self._current_frame_index = None

    def clear(self) -> None:
        with self._lock:
            self._frame_cache.clear()
            self._flow_cache.clear()

    def unlink(self) -> None:
        try:
            self.close()
            self.video_path.unlink(missing_ok=True)
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
    
    def save_frames_as_images(
        self,
        frame_numbers: list[int],
        output_dir: Path | str,
        prefix: str = "frame",
        ext: str = "png",
        save_flow: bool = False,
    ) -> list[Path]:
        """
        Save selected frames as image files.
        
        Args:
            frame_numbers: List of frame numbers to save
            output_dir: Directory to save images to
            prefix: Filename prefix (default: "frame")
            ext: Image extension - png, jpg, bmp (default: "png")
            save_flow: Also save optical flow images (default: False)
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths: list[Path] = []
        
        for frame_num in frame_numbers:
            # Save frame
            frame = self.get_frame(frame_num)
            frame_path = output_dir / f"{prefix}_{frame_num:06d}.{ext}"
            success = imwrite(str(frame_path), frame)
            if success:
                saved_paths.append(frame_path)
                logger.info(f"Saved frame {frame_num} to {frame_path}")
            else:
                logger.error(f"Failed to save frame {frame_num}")
            
            # Save flow if requested
            if save_flow and frame_num > 0:
                try:
                    flow = self.get_flow(frame_num)
                    flow_path = output_dir / f"{prefix}_{frame_num:06d}_flow.{ext}"
                    if imwrite(str(flow_path), flow):
                        saved_paths.append(flow_path)
                        logger.info(f"Saved flow {frame_num} to {flow_path}")
                except Exception as e:
                    logger.error(f"Failed to save flow {frame_num}: {e}")
        
        return saved_paths


async def download_video_cached(
    url: str,
    # _frame_numbers: list[int],  # retained for backward compatibility
    cached_path: Path | None = None,
) -> tuple[str, FrameStore]:
    """
    Download the video once and reuse the cached file across retries.
    When `cached_path` is provided, the file is not re-downloaded.
    The returned Path should be cleaned up by the caller when no longer needed.
    """
    logger.info(f"Downloading video from {url} with cached_path={cached_path}")
    
    # if cached_path is None:
    #     session = await get_async_client()
    #     temp_path: Path | None = None
    #     try:
    #         logger.info(f"Starting download of video from {url}...")
    #         async with session.get(url) as response:
    #             if response.status != 200:
    #                 txt = await response.text()
    #                 raise RuntimeError(
    #                     f"Download failed {response.status}: {txt[:200]}"
    #                 )
    #             logger.info(f"Downloading video to temporary file...")
    #             with NamedTemporaryFile(
    #                 prefix="sv_video_", suffix=".mp4", delete=False
    #             ) as tmp:
    #                 async for chunk in response.content.iter_chunked(1024 * 1024):
    #                     tmp.write(chunk)
    #                 temp_path = Path(tmp.name)
    #     except Exception:
    #         logger.error(f"Error downloading video from {url}")
    #         if temp_path is not None:
    #             temp_path.unlink(missing_ok=True)
    #             logger.info(f"Deleted temporary file {temp_path}")
    #         raise
    #     video_path = temp_path
    # else:
    #     video_path = cached_path
    local_path = Path(url)
    if not local_path.exists():
        raise FileNotFoundError(f"Local video file not found: {local_path}")
    
    video_path = local_path
    name = local_path.name
    # name = url.split("/")[-1]
    logger.info(f"Video name: {name}")
    
    return name, FrameStore(video_path)
