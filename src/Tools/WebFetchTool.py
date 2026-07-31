import json
import logging
import re
import time
from typing import Optional, Dict, Any, ClassVar
from urllib.parse import urlparse

import requests

from src.Tools.BaseTool import BaseTool
from src.config import Config

logger = logging.getLogger(__name__)


class WebFetchTool(BaseTool):
    """Fetches HTTP/HTTPS content from URLs.

    Makes GET or POST requests from the agent_controller service (which has network access).
    The isolated sandbox_worker cannot make network requests.
    Returns structured response with status code, headers, and truncated body.
    """

    url: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    body: Optional[str] = None
    timeout: Optional[int] = 10
    follow_redirects: bool = True

    BLOCKLIST_PATTERNS: ClassVar[list] = [
        r"^127\.",  # 127.0.0.0/8 loopback
        r"^localhost$",
        r"^::1$",  # IPv6 loopback
        r"^0\.0\.0\.0$",  # Default route
        r"^192\.168\.",  # 192.168.0.0/16 private
        r"^10\.",  # 10.0.0.0/8 private
        r"^172\.(1[6-9]|2[0-9]|3[01])\.",  # 172.16.0.0/12 private
        r"^169\.254\.",  # 169.254.0.0/16 link-local
    ]

    def execute(self) -> str:
        """Execute HTTP request with security validation and response truncation."""
        try:
            # Validate and normalize URL
            parsed_url = urlparse(self.url)

            # 1. Scheme validation
            if parsed_url.scheme not in ["http", "https"]:
                return self._error_response(
                    "Invalid URL scheme. Only http and https are allowed."
                )

            # 2. SSRF validation (if enabled)
            if Config.WEBFETCH_BLOCKLIST_INTERNAL_IPS:
                hostname = parsed_url.hostname or ""
                if self._is_internal_ip(hostname):
                    return self._error_response(
                        f"SSRF Blocked: '{hostname}' is an internal IP range. Use public URLs only."
                    )

            # 3. Timeout validation
            # `or` would treat an explicit timeout=0 as falsy and silently
            # substitute the default, letting the "too short" check below
            # never see the real value the caller passed.
            timeout = self.timeout if self.timeout is not None else Config.WEBFETCH_DEFAULT_TIMEOUT
            if timeout > Config.WEBFETCH_MAX_TIMEOUT:
                return self._error_response(
                    f"Timeout {timeout}s exceeds maximum {Config.WEBFETCH_MAX_TIMEOUT}s."
                )
            if timeout < 1:
                return self._error_response("Timeout must be at least 1 second.")

            # 4. Method validation
            if self.method.upper() not in ["GET", "POST"]:
                return self._error_response(
                    "Only GET and POST methods are supported."
                )

            # 5. Build request
            request_headers = {"User-Agent": "mini-claude-code-agent/1.0"}
            if self.headers:
                request_headers.update(self.headers)

            start_time = time.time()

            # 6. Make request
            response = requests.request(
                method=self.method.upper(),
                url=self.url,
                headers=request_headers,
                data=self.body if self.method.upper() == "POST" else None,
                timeout=timeout,
                allow_redirects=self.follow_redirects,
                verify=True,  # Enforce SSL verification
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            # 7. Truncate response if needed
            body = response.text
            truncated = False
            if len(body) > Config.WEBFETCH_MAX_RESPONSE_SIZE:
                body = body[: Config.WEBFETCH_MAX_RESPONSE_SIZE]
                truncated = True

            # 8. Build response
            result = {
                "status_code": response.status_code,
                "status_reason": response.reason,
                "headers": dict(response.headers),
                "body": body,
                "size_bytes": len(response.content),
                "truncated": truncated,
                "timing_ms": elapsed_ms,
            }

            logger.info(
                f"WebFetch {self.method} {self.url}: {response.status_code} in {elapsed_ms}ms"
            )

            return json.dumps(result)

        except requests.exceptions.Timeout:
            return self._error_response(
                f"Request timeout after {timeout}s. The server is slow or unreachable. Try again later or use a different URL."
            )
        except requests.exceptions.ConnectionError as e:
            return self._error_response(
                f"Connection failed: {str(e)[:100]}. Check the URL or network connectivity."
            )
        except requests.exceptions.HTTPError as e:
            return self._error_response(
                f"HTTP error: {str(e)[:100]}. The server returned an error."
            )
        except requests.exceptions.RequestException as e:
            return self._error_response(
                f"Request failed: {str(e)[:100]}. Please try again."
            )
        except Exception as e:
            return self._error_response(
                f"Unexpected error: {str(e)[:100]}. Contact support if this persists."
            )

    def _is_internal_ip(self, hostname: str) -> bool:
        """Check if hostname matches internal IP blocklist patterns."""
        hostname_lower = hostname.lower()
        for pattern in self.BLOCKLIST_PATTERNS:
            if re.match(pattern, hostname_lower):
                return True
        return False

    def _error_response(self, message: str) -> str:
        """Format error as JSON response."""
        result = {
            "status_code": None,
            "error": message,
            "truncated": False,
        }
        return json.dumps(result)
