"""Tests for smartsplit.cli — argument parsing and mode dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestMain:
    def test_default_is_unified(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit"])
        with patch("smartsplit.cli._run_unified") as mock_unified:
            from smartsplit.cli import main

            main()
            mock_unified.assert_called_once_with(8420, 8421, "127.0.0.1", "INFO", "balanced")

    def test_unified_custom_ports(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit", "--port", "9000", "--proxy-port", "9001"])
        with patch("smartsplit.cli._run_unified") as mock_unified:
            from smartsplit.cli import main

            main()
            mock_unified.assert_called_once_with(9000, 9001, "127.0.0.1", "INFO", "balanced")

    def test_api_only(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit", "--api-only"])
        mock_uvicorn = MagicMock()
        mock_app = MagicMock()
        with (
            patch.dict("sys.modules", {"uvicorn": mock_uvicorn}),
            patch("smartsplit.proxy.pipeline.create_app", return_value=mock_app),
        ):
            from smartsplit.cli import main

            main()
            mock_uvicorn.run.assert_called_once()
            call_kwargs = mock_uvicorn.run.call_args
            assert call_kwargs[0][0] is mock_app
            assert call_kwargs[1]["port"] == 8420
            assert call_kwargs[1]["host"] == "127.0.0.1"

    def test_api_only_custom_port(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit", "--api-only", "--port", "9999"])
        mock_uvicorn = MagicMock()
        mock_app = MagicMock()
        with (
            patch.dict("sys.modules", {"uvicorn": mock_uvicorn}),
            patch("smartsplit.proxy.pipeline.create_app", return_value=mock_app),
        ):
            from smartsplit.cli import main

            main()
            assert mock_uvicorn.run.call_args[1]["port"] == 9999

    def test_proxy_only_flag(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit", "--proxy-only"])
        with patch("smartsplit.cli._run_proxy_only") as mock_proxy:
            from smartsplit.cli import main

            main()
            mock_proxy.assert_called_once_with(8420, "127.0.0.1", "INFO", "balanced")

    def test_proxy_legacy_alias(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit", "--proxy"])
        with patch("smartsplit.cli._run_proxy_only") as mock_proxy:
            from smartsplit.cli import main

            main()
            mock_proxy.assert_called_once_with(8420, "127.0.0.1", "INFO", "balanced")

    def test_proxy_with_quality_mode(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit", "--proxy-only", "--mode", "quality"])
        with patch("smartsplit.cli._run_proxy_only") as mock_proxy:
            from smartsplit.cli import main

            main()
            mock_proxy.assert_called_once_with(8420, "127.0.0.1", "INFO", "quality")

    def test_api_only_conflicts_with_proxy_only(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["smartsplit", "--api-only", "--proxy-only"])
        from smartsplit.cli import main

        try:
            main()
        except SystemExit as exc:
            assert exc.code == 2
        err = capsys.readouterr().err
        assert "cannot be combined" in err

    def test_setup_claude_command(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit", "setup-claude"])
        with patch("smartsplit.cli._setup_claude") as mock_setup:
            from smartsplit.cli import main

            main()
            mock_setup.assert_called_once()

    def test_economy_mode_in_api_only(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["smartsplit", "--api-only", "--mode", "economy"])
        mock_uvicorn = MagicMock()
        mock_app = MagicMock()
        with (
            patch.dict("sys.modules", {"uvicorn": mock_uvicorn}),
            patch("smartsplit.proxy.pipeline.create_app", return_value=mock_app) as mock_create,
        ):
            from smartsplit.cli import main

            main()
            mock_create.assert_called_once_with(mode="economy")


class TestRunProxyOnly:
    def test_calls_start_proxy(self):
        with patch("smartsplit.proxy.server.start_proxy") as mock_sp:
            from smartsplit.cli import _run_proxy_only

            _run_proxy_only(8420, "127.0.0.1", "INFO", "balanced")
            mock_sp.assert_called_once_with(port=8420, host="127.0.0.1", log_level="INFO")

    def test_passes_mode_via_env(self, monkeypatch):
        with patch("smartsplit.proxy.server.start_proxy"):
            from smartsplit.cli import _run_proxy_only

            _run_proxy_only(8420, "127.0.0.1", "INFO", "quality")
            import os

            assert os.environ.get("SMARTSPLIT_MODE") == "quality"
        monkeypatch.delenv("SMARTSPLIT_MODE", raising=False)


class TestSetupClaude:
    def test_prints_setup(self, capsys):
        with patch("smartsplit.proxy.server.ensure_certs", return_value="/tmp/ca-cert.pem"):
            from smartsplit.cli import _setup_claude

            _setup_claude()
            output = capsys.readouterr().out
            assert "SmartSplit" in output
            assert "/tmp/ca-cert.pem" in output
