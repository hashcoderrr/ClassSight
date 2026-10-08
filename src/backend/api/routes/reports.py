from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from src.backend.db.database import get_db
from src.backend.services.report_service import (
    export_behavior_report, export_engagement_report, export_attendance_report, export_summary_report
)
from src.backend.api.dependencies import get_current_user
from src.backend.db.models import User

router = APIRouter(prefix="/api/lectures/{id}/reports", tags=["Reports"])

@router.get("/behavior", summary="Export behavior report (JSON or CSV)")
def get_behavior_report(id: int, format: str = Query("json", pattern="^(json|csv)$"), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        data, media_type = export_behavior_report(db, id, current_user, format=format)
        if format.lower() == "csv":
            return Response(content=data, media_type=media_type, headers={"Content-Disposition": f"attachment; filename=behavior_lecture_{id}.csv"})
        return data
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/engagement", summary="Export engagement report (JSON or CSV)")
def get_engagement_report(id: int, format: str = Query("json", pattern="^(json|csv)$"), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        data, media_type = export_engagement_report(db, id, current_user, format=format)
        if format.lower() == "csv":
            return Response(content=data, media_type=media_type, headers={"Content-Disposition": f"attachment; filename=engagement_lecture_{id}.csv"})
        return data
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/attendance", summary="Export attendance report (JSON or CSV)")
def get_attendance_report(id: int, format: str = Query("json", pattern="^(json|csv)$"), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        data, media_type = export_attendance_report(db, id, current_user, format=format)
        if format.lower() == "csv":
            return Response(content=data, media_type=media_type, headers={"Content-Disposition": f"attachment; filename=attendance_lecture_{id}.csv"})
        return data
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

@router.get("/summary", summary="Export lecture summary report (JSON or CSV)")
def get_summary_report(id: int, format: str = Query("json", pattern="^(json|csv)$"), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    try:
        data, media_type = export_summary_report(db, id, current_user, format=format)
        if format.lower() == "csv":
            return Response(content=data, media_type=media_type, headers={"Content-Disposition": f"attachment; filename=summary_lecture_{id}.csv"})
        return data
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
