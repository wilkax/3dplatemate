from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    blob_read_write_token: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()

