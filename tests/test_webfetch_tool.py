import json
import pytest
from unittest.mock import patch, MagicMock

from src.Tools.WebFetchTool import WebFetchTool


class TestWebFetchToolValidation:
    """Test WebFetchTool request validation (SSRF, timeout, method)."""

    def test_invalid_scheme(self):
        """Should reject non-http/https schemes."""
        tool = WebFetchTool(url="ftp://example.com")
        result = json.loads(tool.execute())
        assert result.get("error") is not None
        assert "scheme" in result["error"].lower()

    def test_ssrf_blocklist_localhost(self):
        """Should block localhost SSRF attempts."""
        tool = WebFetchTool(url="http://localhost:8000")
        result = json.loads(tool.execute())
        assert result.get("error") is not None
        assert "SSRF" in result["error"] or "localhost" in result["error"]

    def test_ssrf_blocklist_127_0_0_1(self):
        """Should block 127.0.0.1 SSRF attempts."""
        tool = WebFetchTool(url="http://127.0.0.1:5000")
        result = json.loads(tool.execute())
        assert result.get("error") is not None
        assert "SSRF" in result["error"] or "internal" in result["error"].lower()

    def test_ssrf_blocklist_private_192_168(self):
        """Should block 192.168.* private range."""
        tool = WebFetchTool(url="http://192.168.1.100")
        result = json.loads(tool.execute())
        assert result.get("error") is not None
        assert "SSRF" in result["error"] or "internal" in result["error"].lower()

    def test_ssrf_blocklist_private_10(self):
        """Should block 10.* private range."""
        tool = WebFetchTool(url="http://10.0.0.1")
        result = json.loads(tool.execute())
        assert result.get("error") is not None

    def test_ssrf_blocklist_private_172_16(self):
        """Should block 172.16.*/12 private range."""
        tool = WebFetchTool(url="http://172.16.0.1")
        result = json.loads(tool.execute())
        assert result.get("error") is not None

    def test_timeout_exceeds_maximum(self):
        """Should reject timeouts exceeding max."""
        tool = WebFetchTool(url="http://example.com", timeout=60)
        result = json.loads(tool.execute())
        assert result.get("error") is not None
        assert "timeout" in result["error"].lower()

    def test_timeout_too_short(self):
        """Should reject timeouts < 1 second."""
        tool = WebFetchTool(url="http://example.com", timeout=0)
        result = json.loads(tool.execute())
        assert result.get("error") is not None

    def test_invalid_method(self):
        """Should reject methods other than GET/POST."""
        tool = WebFetchTool(url="http://example.com", method="DELETE")
        result = json.loads(tool.execute())
        assert result.get("error") is not None
        assert "method" in result["error"].lower()


class TestWebFetchToolRequests:
    """Test WebFetchTool actual HTTP requests (mocked)."""

    @patch("src.Tools.WebFetchTool.requests.request")
    def test_successful_get(self, mock_request):
        """Should handle successful GET requests."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.reason = "OK"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"status": "success"}'
        mock_response.content = b'{"status": "success"}'
        mock_request.return_value = mock_response

        tool = WebFetchTool(url="http://example.com")
        result = json.loads(tool.execute())

        assert result["status_code"] == 200
        assert result["status_reason"] == "OK"
        assert '"status": "success"' in result["body"]
        assert result["truncated"] is False

    @patch("src.Tools.WebFetchTool.requests.request")
    def test_successful_post(self, mock_request):
        """Should handle successful POST requests."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.reason = "Created"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"id": 1}'
        mock_response.content = b'{"id": 1}'
        mock_request.return_value = mock_response

        tool = WebFetchTool(
            url="http://example.com/api",
            method="POST",
            body='{"name": "test"}'
        )
        result = json.loads(tool.execute())

        assert result["status_code"] == 201
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["data"] == '{"name": "test"}'

    @patch("src.Tools.WebFetchTool.requests.request")
    def test_response_truncation(self, mock_request):
        """Should truncate responses exceeding max size."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.reason = "OK"
        mock_response.headers = {}
        # Create response larger than 102400 bytes
        mock_response.text = "x" * 150000
        mock_response.content = b"x" * 150000
        mock_request.return_value = mock_response

        tool = WebFetchTool(url="http://example.com")
        result = json.loads(tool.execute())

        assert result["truncated"] is True
        assert len(result["body"]) == 102400  # Should be truncated to max size

    @patch("src.Tools.WebFetchTool.requests.request")
    def test_ssl_verification_enabled(self, mock_request):
        """Should enforce SSL verification."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_response.text = "ok"
        mock_response.content = b"ok"
        mock_request.return_value = mock_response

        tool = WebFetchTool(url="https://example.com")
        tool.execute()

        # Verify SSL verification was enabled
        args, kwargs = mock_request.call_args
        assert kwargs["verify"] is True

    @patch("src.Tools.WebFetchTool.requests.request")
    def test_user_agent_header(self, mock_request):
        """Should set User-Agent header."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_response.text = "ok"
        mock_response.content = b"ok"
        mock_request.return_value = mock_response

        tool = WebFetchTool(url="http://example.com")
        tool.execute()

        args, kwargs = mock_request.call_args
        assert "User-Agent" in kwargs["headers"]
        assert "mfca-agent" in kwargs["headers"]["User-Agent"]

    @patch("src.Tools.WebFetchTool.requests.request")
    def test_custom_headers(self, mock_request):
        """Should merge custom headers with defaults."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.reason = "OK"
        mock_response.headers = {}
        mock_response.text = "ok"
        mock_response.content = b"ok"
        mock_request.return_value = mock_response

        tool = WebFetchTool(
            url="http://example.com",
            headers={"Authorization": "Bearer token123"}
        )
        tool.execute()

        args, kwargs = mock_request.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer token123"
        assert "User-Agent" in kwargs["headers"]


class TestWebFetchToolErrorHandling:
    """Test WebFetchTool error handling."""

    @patch("src.Tools.WebFetchTool.requests.request")
    def test_timeout_error(self, mock_request):
        """Should handle request timeouts gracefully."""
        import requests
        mock_request.side_effect = requests.exceptions.Timeout("Connection timed out")

        tool = WebFetchTool(url="http://example.com", timeout=5)
        result = json.loads(tool.execute())

        assert result.get("error") is not None
        assert "timeout" in result["error"].lower()

    @patch("src.Tools.WebFetchTool.requests.request")
    def test_connection_error(self, mock_request):
        """Should handle connection errors gracefully."""
        import requests
        mock_request.side_effect = requests.exceptions.ConnectionError("Network unreachable")

        tool = WebFetchTool(url="http://example.com")
        result = json.loads(tool.execute())

        assert result.get("error") is not None
        assert "connection" in result["error"].lower()

    @patch("src.Tools.WebFetchTool.requests.request")
    def test_http_error(self, mock_request):
        """Should handle HTTP errors gracefully."""
        import requests
        mock_request.side_effect = requests.exceptions.HTTPError("HTTP Error")

        tool = WebFetchTool(url="http://example.com")
        result = json.loads(tool.execute())

        assert result.get("error") is not None

    @patch("src.Tools.WebFetchTool.requests.request")
    def test_generic_exception(self, mock_request):
        """Should handle unexpected exceptions."""
        mock_request.side_effect = Exception("Unexpected error")

        tool = WebFetchTool(url="http://example.com")
        result = json.loads(tool.execute())

        assert result.get("error") is not None


class TestWebFetchToolSSRFBlocklist:
    """Test SSRF blocklist patterns comprehensively."""

    def test_is_internal_ip_127_range(self):
        """Test 127.x.x.x loopback range."""
        tool = WebFetchTool(url="http://example.com")
        assert tool._is_internal_ip("127.0.0.1") is True
        assert tool._is_internal_ip("127.1.1.1") is True
        assert tool._is_internal_ip("127.255.255.255") is True

    def test_is_internal_ip_192_168_range(self):
        """Test 192.168.x.x private range."""
        tool = WebFetchTool(url="http://example.com")
        assert tool._is_internal_ip("192.168.0.1") is True
        assert tool._is_internal_ip("192.168.1.254") is True
        assert tool._is_internal_ip("192.169.0.1") is False  # Outside range

    def test_is_internal_ip_10_range(self):
        """Test 10.x.x.x private range."""
        tool = WebFetchTool(url="http://example.com")
        assert tool._is_internal_ip("10.0.0.1") is True
        assert tool._is_internal_ip("10.255.255.255") is True
        assert tool._is_internal_ip("11.0.0.1") is False

    def test_is_internal_ip_172_16_range(self):
        """Test 172.16.x.x to 172.31.x.x private range."""
        tool = WebFetchTool(url="http://example.com")
        assert tool._is_internal_ip("172.16.0.1") is True
        assert tool._is_internal_ip("172.31.255.255") is True
        assert tool._is_internal_ip("172.15.0.1") is False  # Below range
        assert tool._is_internal_ip("172.32.0.1") is False  # Above range

    def test_is_internal_ip_localhost(self):
        """Test localhost variations."""
        tool = WebFetchTool(url="http://example.com")
        assert tool._is_internal_ip("localhost") is True
        assert tool._is_internal_ip("LOCALHOST") is True

    def test_is_internal_ip_ipv6_loopback(self):
        """Test IPv6 loopback."""
        tool = WebFetchTool(url="http://example.com")
        assert tool._is_internal_ip("::1") is True

    def test_is_internal_ip_public(self):
        """Test that public IPs are not blocked."""
        tool = WebFetchTool(url="http://example.com")
        assert tool._is_internal_ip("8.8.8.8") is False
        assert tool._is_internal_ip("1.1.1.1") is False
        assert tool._is_internal_ip("example.com") is False


class TestWebFetchToolResponseStructure:
    """Test response structure and fields."""

    @patch("src.Tools.WebFetchTool.requests.request")
    def test_response_includes_all_fields(self, mock_request):
        """Should include all required fields in response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.reason = "OK"
        mock_response.headers = {"x-custom": "value"}
        mock_response.text = "test body"
        mock_response.content = b"test body"
        mock_request.return_value = mock_response

        tool = WebFetchTool(url="http://example.com")
        result = json.loads(tool.execute())

        assert "status_code" in result
        assert "status_reason" in result
        assert "headers" in result
        assert "body" in result
        assert "size_bytes" in result
        assert "truncated" in result
        assert "timing_ms" in result

    def test_error_response_structure(self):
        """Should include error field in error responses."""
        tool = WebFetchTool(url="invalid-scheme://example.com")
        result = json.loads(tool.execute())

        assert "error" in result
        assert result["status_code"] is None
