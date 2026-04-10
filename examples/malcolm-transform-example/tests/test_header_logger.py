"""Tests for the header_logger example transform."""

import logging

import pytest

from malcolm_transform_example import HeaderLoggerTransform, create


class TestHeaderLoggerTransform:
    def test_name(self):
        t = HeaderLoggerTransform()
        assert t.name == "header_logger"

    def test_request_is_pass_through(self):
        t = HeaderLoggerTransform()
        body = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4"}
        result = t.transform_request(body)
        assert result is body

    def test_response_is_pass_through(self):
        t = HeaderLoggerTransform()
        body = {"content": "hello"}
        result = t.transform_response(body, model="gpt-4o")
        assert result is body

    def test_stream_line_returns_line_unchanged(self):
        t = HeaderLoggerTransform()
        state: dict = {}
        result = t.transform_stream_line('data: {"x":1}', state)
        assert result == ['data: {"x":1}']

    def test_stream_line_counts_in_state(self):
        t = HeaderLoggerTransform()
        state: dict = {}
        t.transform_stream_line("a", state)
        t.transform_stream_line("b", state)
        t.transform_stream_line("c", state)
        assert state["lines_seen"] == 3

    def test_rewrite_path_pass_through(self):
        t = HeaderLoggerTransform()
        assert t.rewrite_path("/v1/messages") == "/v1/messages"

    def test_request_logs_prefix_and_size(self, caplog):
        t = HeaderLoggerTransform()
        with caplog.at_level(logging.INFO, logger="malcolm_transform_example"):
            t.transform_request({"a": 1, "b": 2})
        messages = [r.message for r in caplog.records]
        assert any("[header_logger]" in m for m in messages)
        assert any("size=" in m for m in messages)

    def test_request_logs_sorted_keys(self, caplog):
        t = HeaderLoggerTransform()
        with caplog.at_level(logging.INFO, logger="malcolm_transform_example"):
            t.transform_request({"zebra": 1, "alpha": 2})
        messages = [r.message for r in caplog.records]
        # keys should appear sorted so logs are stable
        request_line = next(m for m in messages if "request" in m)
        assert request_line.index("alpha") < request_line.index("zebra")

    def test_response_logs_model(self, caplog):
        t = HeaderLoggerTransform()
        with caplog.at_level(logging.INFO, logger="malcolm_transform_example"):
            t.transform_response({"x": 1}, model="gpt-4o")
        assert any("gpt-4o" in r.message for r in caplog.records)

    def test_custom_prefix(self, caplog):
        t = HeaderLoggerTransform(prefix="[custom]")
        with caplog.at_level(logging.INFO, logger="malcolm_transform_example"):
            t.transform_request({})
        assert any("[custom]" in r.message for r in caplog.records)
        assert not any("[header_logger]" in r.message for r in caplog.records)

    def test_custom_logger_name_routes_elsewhere(self, caplog):
        t = HeaderLoggerTransform(logger_name="alt.logger")
        with caplog.at_level(logging.INFO, logger="alt.logger"):
            t.transform_request({})
        assert any("[header_logger]" in r.message for r in caplog.records)


class TestCreateFactory:
    def test_default_config(self):
        t = create({})
        assert isinstance(t, HeaderLoggerTransform)
        assert t.name == "header_logger"

    def test_empty_config_uses_default_prefix(self, caplog):
        t = create({})
        with caplog.at_level(logging.INFO, logger="malcolm_transform_example"):
            t.transform_request({})
        assert any("[header_logger]" in r.message for r in caplog.records)

    def test_custom_prefix_via_config(self, caplog):
        t = create({"prefix": "[MY-PREFIX]"})
        with caplog.at_level(logging.INFO, logger="malcolm_transform_example"):
            t.transform_request({})
        assert any("[MY-PREFIX]" in r.message for r in caplog.records)

    def test_custom_logger_name_via_config(self, caplog):
        t = create({"logger_name": "custom.logger"})
        with caplog.at_level(logging.INFO, logger="custom.logger"):
            t.transform_request({})
        assert any("[header_logger]" in r.message for r in caplog.records)

    def test_factory_ignores_unknown_keys(self):
        # Unknown keys should not raise — forward-compat for older configs
        t = create({"prefix": "[x]", "unknown_future_key": True})
        assert t.name == "header_logger"
