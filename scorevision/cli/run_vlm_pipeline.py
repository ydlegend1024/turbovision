from enum import Enum
from logging import getLogger
from time import monotonic

from huggingface_hub import HfApi

from matplotlib.pyplot import box
from scorevision.utils.challenges import prepare_challenge_payload
from scorevision.vlm_pipeline.vlm_annotator import (
    generate_annotations_for_select_frames,
)
from scorevision.utils.predict import call_miner_model_on_chutes
from scorevision.utils.data_models import SVChallenge, SVPredictResult, SVRunOutput
from scorevision.utils.chutes_helpers import (
    get_chute_slug_and_id,
)
from scorevision.utils.async_clients import get_async_client
from scorevision.utils.evaluate import post_vlm_ranking
from scorevision.vlm_pipeline.non_vlm_scoring.smoothness import (
    filter_low_quality_pseudo_gt_annotations,
)
from scorevision.vlm_pipeline.non_vlm_scoring.objects import (
    _extract_boxes_labels,
)
from scorevision.utils.data_models import SVEvaluation
from scorevision.vlm_pipeline.domain_specific_schemas.challenge_types import (
    ChallengeType,
)
from scorevision.utils.huggingface_helpers import get_huggingface_repo_revision
from scorevision.utils.settings import get_settings
from scorevision.utils.image_processing import save_annotated_frames

from cv2 import (
    imwrite, imread
)

import glob
import os

logger = getLogger(__name__)


async def run_vlm_pipeline_once_for_single_miner(
    hf_revision: str | None,
) -> SVEvaluation:
    """Run a single miner on the VLM pipeline off-chain
    NOTE: This flow should match the flow in the runner"""
    challenge_data = {
        "task_id": "0",
        "video_url": "/work/bittensor/sn44/turbovision/sample_videos/sv_video_1.mp4",
    }
    logger.info(f"Challenge data from API: {challenge_data}")
    if not hf_revision:
        settings = get_settings()
        hf_api = HfApi(token=settings.HUGGINGFACE_API_KEY.get_secret_value())
        hf_revision = await get_huggingface_repo_revision(hf_api=hf_api)

    payload, frame_numbers, frames, flows, frame_store = (
        await prepare_challenge_payload(challenge=challenge_data)
    )
    
    frame_numbers = [0]
    test_img = imread("/work/bittensor/sn44/turbovision/sample_videos/test.jpg")
    frames = [test_img]
    flows = [test_img]
    
    if not payload:
        raise Exception("Failed to prepare payload from challenge.")

    # chute_slug, chute_id = await get_chute_slug_and_id(revision=hf_revision)
    # if not chute_slug:
    #     raise Exception("Failed to fetch chute slug")

    # logger.info("Calling model from chutes API")

    miner_output = await call_miner_model_on_chutes(
        # slug=chute_slug,
        # chute_id=chute_id,
        payload=payload,
        frames=frames,
        frame_numbers=frame_numbers,
    )
    
    # miner_output = SVRunOutput(
    #     success=False,
    #     latency_ms=1000.0,
    #     predictions=None,
    #     error=None,
    #     model=None,
    # )
    logger.info(f"Miner: {miner_output.model} | Success: {miner_output.success}")

    challenge = SVChallenge(
        env="SVEnv",
        payload=payload,
        meta={},
        prompt="ScoreVision video task mock-challenge",
        challenge_id="0",
        frame_numbers=frame_numbers,
        frames=frames,
        dense_optical_flow_frames=flows,
        challenge_type=ChallengeType.FOOTBALL,
    )
    # logger.info(f"Challenge: {challenge}")
    t0 = monotonic()
    pseudo_gt_annotations = await generate_annotations_for_select_frames(
        video_name=challenge.challenge_id,
        frames=challenge.frames,
        flow_frames=challenge.dense_optical_flow_frames,
        frame_numbers=challenge.frame_numbers,
    )
    logger.info(f"{len(pseudo_gt_annotations)} Pseudo GT annotations generated")
    pseudo_gt_annotations = filter_low_quality_pseudo_gt_annotations(
        annotations=pseudo_gt_annotations
    )
    logger.info(f"/n/nFiltering low-quality Pseudo GT annotations took {monotonic() - t0:.2f}s")
    logger.info(
        f"{len(pseudo_gt_annotations)} Pseudo GT annotations had sufficient quality"
    )
    
    # Save annotated frames with bounding boxes for visualization
    output_dir = "/work/bittensor/sn44/turbovision/annotated_frames"
    for file_path in glob.glob(output_dir + "/*.png"):
        os.remove(file_path)

    # saved_files = save_annotated_frames(
    #     pseudo_gt_annotations=pseudo_gt_annotations,
    #     output_dir=output_dir,
    #     prefix="pseudo_gt",
    # )
    # logger.info(f"Saved {len(saved_files)} annotated frames to {output_dir}")

    evaluation = post_vlm_ranking(
        payload=payload,
        miner_run=miner_output,
        challenge=challenge,
        pseudo_gt_annotations=pseudo_gt_annotations,
        frame_store=frame_store,
    )
    # frame_store.unlink()
    logger.info(f"\n\n✅Evaluation: {evaluation}")
    return evaluation

# =======================
# Enums
# =======================
class Person(Enum):
    BALL = "ball"
    GOALIE = "goalkeeper"
    PLAYER = "player"
    REFEREE = "referee"
    
LABEL_CONFIG_FOOTBALL = {
    "ball": 0,
    "goalkeeper": 1,
    "referee": 2,
    "player1": 3,
    "player2": 4,
}

TEAM_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 165, 255),
    "purple": (128, 0, 128),
    "pink": (203, 192, 255),
    "default": (128, 128, 128),
}

async def generate_train_data(
) -> bool:
    """Run a single miner on the VLM pipeline off-chain
    NOTE: This flow should match the flow in the runner"""
    
    logger.info(f"\n\n Generating Train Data...\n\n")
    
    # challenge_data = {
    #     "task_id": "0",
    #     "video_url": "/work/bittensor/sn44/turbovision/sample_videos/sv_video_1.mp4",
    # }
    # logger.info(f"Challenge data from API: {challenge_data}")
    # if not hf_revision:
    #     settings = get_settings()
    #     hf_api = HfApi(token=settings.HUGGINGFACE_API_KEY.get_secret_value())
    #     hf_revision = await get_huggingface_repo_revision(hf_api=hf_api)

    # frame_numbers = [0]
    # test_img = imread("/work/bittensor/sn44/turbovision/sample_videos/test.jpg")
    # frames = [test_img]
    # flows = [test_img]
    
    # image_folder = '/work/bittensor/sn44/turbovision/sample_videos'
    image_folder = '/work/bittensor/sn44/videoCapture/dataset_00000-01500/images'
    label_folder = '//work/bittensor/sn44/videoCapture/dataset_00000-01500/labels_with_team'
    os.makedirs(label_folder, exist_ok=True)
    
    for filename in sorted(os.listdir(image_folder)):
        base, _ = os.path.splitext(filename)
        # validate_extensions = ['.jpg', '.jpeg', '.png']
        # if not any(filename.lower().endswith(ext) for ext in validate_extensions):
        #     continue
        
        label_filename = base + ".txt"
        if os.path.exists(os.path.join(label_folder, label_filename)):
            continue
        
        img = imread(os.path.join(image_folder, filename))
        # img = imread("/work/bittensor/sn44/turbovision/sample_videos/test.jpg")
        
        img_width, img_height = img.shape[1], img.shape[0]
        print(f"\n\nProcessing image: {filename} (width={img_width}, height={img_height})\n\n")
        
        t0 = monotonic()
        pseudo_gt_annotations = await generate_annotations_for_select_frames(
            video_name='video',
            frames=[img],
            flow_frames=[img],
            frame_numbers=[0],
        )
        logger.info(f"{len(pseudo_gt_annotations)} Pseudo GT annotations generated")
        if not pseudo_gt_annotations or len(pseudo_gt_annotations) == 0:
            logger.info(f"No high-quality Pseudo GT annotations for image: {filename}")
            open(os.path.join(label_folder, label_filename), "w").close()
            continue
        
        pseudo_gt_annotations = filter_low_quality_pseudo_gt_annotations(
            annotations=pseudo_gt_annotations
        )


        team_map = {}
        def get_team(color_name):
            if color_name not in team_map:
                team_map[color_name] = len(team_map)+3  # Starting from 3 to avoid conflict with other class IDs
            return team_map[color_name]

        log_file = "/work/bittensor/sn44/turbovision/log.txt"
        with open(os.path.join(label_folder, label_filename), "w") as f:
            for bbox in pseudo_gt_annotations[0].annotation.bboxes:
                x_min, y_min, x_max, y_max = bbox.bbox_2d
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                label_str = str(bbox.label.value).lower() if hasattr(bbox.label, 'value') else str(bbox.label).lower()
                # logger.info(f"\n\nAnnotation: {label_str} | BBox: {bbox}\n")
                class_id = LABEL_CONFIG_FOOTBALL.get(label_str, -1)
                if label_str == "player":
                    cluster_str = str(bbox.cluster_id.value).lower() if hasattr(bbox.cluster_id, 'value') else str(bbox.cluster_id).lower()
                    class_id = get_team(cluster_str)
                    # logger.info(f"Cluster ID: {cluster_str}, {class_id}\n")
                
                if class_id == -1 or class_id > 4:
                    skip_msg = f"Skipping unknown label: {label_str} | class_id: {class_id} | File: {filename}\n"
                    logger.info(skip_msg)
                    with open(log_file, "a") as log:
                        log.write(skip_msg)
                    continue
            
                # logger.info(f"\n{class_id} {min(x_center/img_width, 1)} {min(y_center/img_height, 1)} {min(width/img_width, 1)} {min(height/img_height, 1)}\n")
                f.write(f"{class_id} {min(x_center/img_width, 1)} {min(y_center/img_height, 1)} {min(width/img_width, 1)} {min(height/img_height, 1)}\n")

        logger.info(f"/n/nFiltering low-quality Pseudo GT annotations took {monotonic() - t0:.2f}s")
        # logger.info(
        #     f"{len(pseudo_gt_annotations)} Pseudo GT annotations had sufficient quality"
        # )
        
        # Save annotated frames with bounding boxes for visualization
        # output_dir = "/work/bittensor/sn44/turbovision/annotated_frames"
        # for file_path in glob.glob(output_dir + "/*.png"):
            # os.remove(file_path)

        # saved_files = save_annotated_frames(
        #     pseudo_gt_annotations=pseudo_gt_annotations,
        #     output_dir=output_dir,
        #     prefix="pseudo_gt",
        # )
        # logger.info(f"Saved {len(saved_files)} annotated frames to {output_dir}")

        # break

        # frame_store.unlink()
    
    return True
