from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    claude_model: str = "claude-opus-4-5"

    class Config:
        env_file = ".env"


settings = Settings()

