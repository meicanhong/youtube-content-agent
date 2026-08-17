from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="ignore")

    editorial_provider: Literal["mimo", "openai"] = "openai"
    mimo_api_key: str | None = None
    mimo_model: str = "mimo-v2.5"
    mimo_base_url: str = "https://api.xiaomimimo.com/v1"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.6-luna"
    openai_base_url: str = "http://47.84.236.4:8080/v1"
    youtube_api_key: str | None = None
    yt_dlp_bin: str = "yt-dlp"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    yt_dlp_cookies_from_browser: str | None = None
