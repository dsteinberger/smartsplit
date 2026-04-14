"""Tests for smartsplit.cli — argument parsing and mode dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestMain:
    def test_default_mode_is_balanced(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit"])
        mock_uvicorn = MagicMock()
        mock_app = MagicMock()
        with (
            patch.dict("sys.modules", {"uvicorn": mock_uvicorn}),
            patch("smartsplit.pipeline.create_app", return_value=mock_app),
        ):
            from smartsplit.cli import main

            main()
            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args
            assert call_kwargs[0][0] is mock_app
            assert call_kwargs[1]["port"] == 8420
            assert call_kwargs[1]["host"] == "127.0.0.1"

    def test_custom_port(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit", "--port", "9999"])
        mock_uvicorn = MagicMock()
        mock_app = MagicMock()
        with (
            patch.dict("sys.modules", {"uvicorn": mock_uvicorn}),
            patch("smartsplit.pipeline.create_app", return_value=mock_app),
        ):
            from smartsplit.cli import main

            main()
            assert mock_uvicorn.run.call_args[1]["port"] == 9999

    def test_proxy_mode(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit", "--mode", "proxy"])
        with patch("smartsplit.cli._run_proxy_mode") as mock_proxy:
            from smartsplit.cli import main

            main()
            mock_proxy.assert_called_once_with(8420, "127.0.0.1", "INFO")

    def test_proxy_mitm_mode(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit", "--mode", "proxy-mitm"])
        with patch("smartsplit.cli._run_proxy_mitmproxy") as mock_mitm:
            from smartsplit.cli import main

            main()
            mock_mitm.assert_called_once_with(8420, "INFO")

    def test_setup_claude_command(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit", "setup-claude"])
        with patch("smartsplit.cli._setup_claude") as mock_setup:
            from smartsplit.cli import main

            main()
            mock_setup.assert_called_once()

    def test_economy_mode(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit", "--mode", "economy"])
        mock_uvicorn = MagicMock()
        mock_app = MagicMock()
        with (
            patch.dict("sys.modules", {"uvicorn": mock_uvicorn}),
            patch("smartsplit.pipeline.create_app", return_value=mock_app) as mock_create,
        ):
            from smartsplit.cli import main

            main()
            mock_create.assert_called_once_with(mode="economy")


class TestRunProxyMode:
    def test_calls_start_proxy(self):
        with patch("smartsplit.proxy.start_proxy") as mock_sp:
            from smartsplit.cli import _run_proxy_mode

            _run_proxy_mode(8420, "127.0.0.1", "INFO")
            mock_sp.assert_called_once_with(port=8420, host="127.0.0.1", log_level="INFO")


class TestRunProxyMitmproxy:
    def test_missing_mitmproxy_raises(self):
        with patch.dict("sys.modules", {"mitmproxy": None, "mitmproxy.tools": None, "mitmproxy.tools.main": None}):
            from smartsplit.cli import _run_proxy_mitmproxy

            with pytest.raises(SystemExit):
                _run_proxy_mitmproxy(8420, "INFO")


class TestSetupClaude:
    def test_prints_setup(self, capsys):
        with patch("smartsplit.proxy.ensure_certs", return_value="/tmp/ca-cert.pem"):
            from smartsplit.cli import _setup_claude

            _setup_claude()
            output = capsys.readouterr().out
            assert "SmartSplit" in output
            assert "/tmp/ca-cert.pem" in output
