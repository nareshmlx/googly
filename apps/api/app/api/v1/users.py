from fastapi import APIRouter, Depends

from app.core.auth import get_current_user, verify_internal_token
from app.models.schemas import UserProfile

router = APIRouter(dependencies=[Depends(verify_internal_token)])


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    return UserProfile(user_id=current_user["user_id"], tier=current_user.get("tier", "free"))


@router.get("/preferences")
async def get_preferences(current_user: dict = Depends(get_current_user)):
    return {"user_id": current_user["user_id"], "preferences": {}}
