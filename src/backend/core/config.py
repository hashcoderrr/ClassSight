import secrets

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "ClassSight AI"
    DATABASE_URL: str = "sqlite:///./classsight.db"
    JWT_SECRET_KEY: str = secrets.token_urlsafe(32)
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    ENVIRONMENT: str = "development"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MIN_ATTENDANCE_DURATION: float = 30.0  # seconds visible to confirm attendance
    MIN_ATTENDANCE_OBSERVATIONS: int = 2   # min recognition/track observations

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

settings = Settings()
