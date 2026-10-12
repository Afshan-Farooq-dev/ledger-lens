"""Read secrets only for authorized calls; no secret-bearing repr."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import dotenv_values

MODEL = 'gemini-3.1-flash-lite'
FREE_MODELS = (MODEL, 'gemini-3.5-flash-lite')

@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = MODEL
    provider: str = 'google'

    def __post_init__(self):
        if self.model not in FREE_MODELS or self.provider != 'google':
            raise ValueError('Only google gemini-3.1-flash-lite or gemini-3.5-flash-lite permitted; no substitution')
        if not self.api_key or self.api_key == 'your-api-key-here':
            raise ValueError('Set GEMINI_API_KEY (or API_KEY) in .env')

    @classmethod
    def load(cls, path=Path('.env')):
        values = dotenv_values(path)
        def get(name, default=None):
            return os.environ.get(name, values.get(name, default))
        return cls(api_key=get('GEMINI_API_KEY') or get('API_KEY') or '',
                   model=get('MODEL_NAME', MODEL), provider=get('MODEL_PROVIDER', 'google'))
