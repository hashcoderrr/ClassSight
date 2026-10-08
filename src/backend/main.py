import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.backend.core.config import settings
from src.backend.api.routes import (
    health, auth, users, students, classrooms,
    lectures, attendance, behavior, engagement, processing, reports
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("classsight.backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ClassSight AI Backend initialized successfully.")
    yield
    logger.info("ClassSight AI Backend shutting down.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API and Persistence Platform for ClassSight AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        logger.error(f"Unhandled error handling {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

# Register Routers
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(students.router)
app.include_router(classrooms.router)
app.include_router(lectures.router)
app.include_router(attendance.router)
app.include_router(behavior.router)
app.include_router(engagement.router)
app.include_router(processing.router)
app.include_router(reports.router)
