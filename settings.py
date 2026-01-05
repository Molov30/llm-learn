from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_provider: str = Field()
    api_key: str = Field()
    tavily_api_key: str = Field()
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def provider_base_url(self) -> str:
        return f"https://{self.api_provider}/v1"


settings = Settings()  # pyright: ignore[reportCallIssue]
