from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.analysis_service import analysis_service

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


class RunAnalysisRequest(BaseModel):
    video_path: str
    inspection_type: Literal["Crack", "Corrosion"] = "Crack"
    corrosion_env: Literal["Wet", "Dry"] = "Wet"
    gsd: float = 0.5436
    frame_stride: int = 1
    inference_width: int = 0


@router.get("/status")
async def analysis_status():
    return analysis_service.get_state()


@router.post("/run")
async def run_analysis(body: RunAnalysisRequest):
    return analysis_service.run(
        video_path=body.video_path,
        inspection_type=body.inspection_type,
        corrosion_env=body.corrosion_env,
        gsd=body.gsd,
        frame_stride=body.frame_stride,
        inference_width=body.inference_width,
    )


@router.post("/stop")
async def stop_analysis():
    return analysis_service.stop()
