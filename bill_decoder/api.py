"""Bounded approved-model calls, with secret-safe errors and explicit pacing."""
import json
import logging
import random
import time
import httpx
from google import genai
from google.genai import errors, types
from .config import MODEL, FREE_MODELS

LOG = logging.getLogger('bill_decoder')

class APIProblem(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def event(name, **fields):
    LOG.info(json.dumps({'event': name, **fields}, ensure_ascii=False))

class Gemini:
    def __init__(self, settings, attempts=3, interval=4.1, sleeper=time.sleep, clock=time.monotonic):
        if settings.model not in FREE_MODELS or settings.provider != 'google':
            raise ValueError('Unapproved model/provider for this phase')
        self.settings, self.attempts, self.interval = settings, attempts, interval
        self.sleep, self.clock, self.last_request = sleeper, clock, None
        self.calls = 0
        self.client = genai.Client(api_key=settings.api_key, vertexai=False,
            http_options=types.HttpOptions(timeout=90000, retry_options=types.HttpRetryOptions(attempts=1)))

    def close(self):
        self.client.close()

    def _message(self, exc):
        # The SDK APIError.message is a service diagnostic, not request headers.
        # Do not print raw exceptions, URLs, response bodies or validation inputs.
        message = str(getattr(exc, 'message', 'API request failed'))
        message = message.replace(self.settings.api_key, '[REDACTED]')
        import re
        message = re.sub(r'AIza[A-Za-z0-9_-]+', '[REDACTED]', message)
        message = re.sub(r'https?://\S+', '[URL omitted]', message)
        return message[:900]

    def generate(self, contents, schema=None):
        config = types.GenerateContentConfig(temperature=0, max_output_tokens=10000 if schema else 64,
                response_mime_type='application/json' if schema else 'text/plain',
                response_json_schema=schema)
        for attempt in range(1, self.attempts + 1):
            if self.last_request is not None:
                self.sleep(max(0, self.interval - (self.clock() - self.last_request)))
            self.last_request = self.clock()
            self.calls += 1
            event('api_call', model=self.settings.model, attempt=attempt)
            try:
                response = self.client.models.generate_content(model=self.settings.model, contents=contents, config=config)
                if not response.text:
                    raise APIProblem('EMPTY_RESPONSE', 'Model returned no text (blocked or empty response)')
                usage = response.usage_metadata
                return response.text, {
                    'prompt_tokens': getattr(usage, 'prompt_token_count', None),
                    'output_tokens': getattr(usage, 'candidates_token_count', None),
                    'response_model_version': getattr(response, 'model_version', None),
                }
            except errors.APIError as exc:
                code = exc.code
                if code not in (408, 429, 500, 502, 503, 504) or attempt == self.attempts:
                    raise APIProblem(code, self._message(exc)) from None
                wait = min(30, 2 ** attempt + random.uniform(0, 1))
                # Some transports expose Retry-After on the response.
                raw_response = getattr(exc, 'response', None)
                headers = getattr(raw_response, 'headers', {}) or {}
                try:
                    wait = min(30, max(wait, float(headers.get('retry-after', 0))))
                except (TypeError, ValueError):
                    pass
                event('api_retry', code=code, delay_seconds=round(wait, 2))
                self.sleep(wait)
            except (httpx.TransportError, TimeoutError) as exc:
                if attempt == self.attempts:
                    raise APIProblem('NETWORK', 'Transport failure: ' + type(exc).__name__) from None
                self.sleep(min(30, 2 ** attempt))
        raise APIProblem('INTERNAL', 'Retry loop exhausted')

    def preflight(self):
        text, usage = self.generate('Connectivity check. Reply with exactly OK.')
        if text.strip() != 'OK':
            raise APIProblem('UNEXPECTED_RESPONSE', 'Model responded but did not return the requested OK')
        return {'status': 'connected', 'model': self.settings.model, 'calls': self.calls, 'usage': usage}
