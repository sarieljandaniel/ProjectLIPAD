from __future__ import annotations

from fastapi import APIRouter

from backend.services.results_service import clear_results, load_results, repair_guidance

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("")
async def get_results():
    return load_results()


@router.get("/repair")
async def get_repair_guidance():
    return repair_guidance()


@router.delete("")
async def delete_results():
    return clear_results()
