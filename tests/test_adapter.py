"""Adapter contract tests with synthetic events and an isolated SQLite database.

No AstrBot installation, model calls, or user's running data are needed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib.util
import itertools
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


@dataclass
class Plain:
    text: str


@dataclass
class TextPart:
    text: str
    is_temp: bool = False

    def mark_as_temp(self):
        self.is_temp = True
        return self


class Star:
    def __init__(self, context):
        self.context = context


class StarTools:
    directory = Path(".")

    @classmethod
    def get_data_dir(cls, _name):
        return cls.directory


class CustomFilter:
    def __init__(self, *args, **kwargs):
        pass


def _decorator(*_args, **_kwargs):
    return lambda function: function


def _modules():
    modules = {
        name: ModuleType(name)
        for name in (
            "astrbot", "astrbot.api", "astrbot.api.event",
            "astrbot.api.event.filter",
            "astrbot.api.message_components", "astrbot.api.star",
            "astrbot.core", "astrbot.core.agent", "astrbot.core.agent.message",
            "astrbot.core.star", "astrbot.core.star.session_llm_manager",
            "astrbot.core.star.session_plugin_manager",
        )
    }
    modules["astrbot.api"].logger = logging.getLogger("directional-memory-test")
    modules["astrbot.api.event"].AstrMessageEvent = object
    filters = modules["astrbot.api.event.filter"]
    filters.CustomFilter = CustomFilter
    filters.EventMessageType = SimpleNamespace(ALL="all")
    filters.event_message_type = _decorator
    filters.on_llm_request = _decorator
    filters.command = _decorator
    filters.custom_filter = _decorator
    modules["astrbot.api.event"].filter = filters
    modules["astrbot.api.message_components"].Plain = Plain
    modules["astrbot.api.star"].Context = object
    modules["astrbot.api.star"].Star = Star
    modules["astrbot.api.star"].StarTools = StarTools
    modules["astrbot.core.agent.message"].TextPart = TextPart
    modules["astrbot.core.star.session_llm_manager"].SessionServiceManager = SimpleNamespace(
        is_session_enabled=AsyncMock(return_value=True),
    )
    modules["astrbot.core.star.session_plugin_manager"].SessionPluginManager = SimpleNamespace(
        is_plugin_enabled_for_session=AsyncMock(return_value=True),
    )
    return modules


class Event:
    _ids = itertools.count()

    def __init__(
        self, text="", *, user="111", group=None, private=None,
        platform="aiocqhttp", bot="999", parts=None, extras=None,
    ):
        self.text = text
        self.user = user
        self.group = group
        self.private = group is None if private is None else private
        self.platform = platform
        self.bot = bot
        self.extras = dict(extras or {})
        self.parts = [Plain(text)] if parts is None else parts
        self.message_obj = SimpleNamespace(
            message_id="synthetic-" + str(next(self._ids)), message=self.parts,
        )
        self.unified_msg_origin = "untrusted-origin-not-used-for-identity"
        self.is_wake = False
        self.is_at_or_wake_command = False

    def get_platform_name(self):
        return self.platform

    def get_platform_id(self):
        return "qq-test"

    def get_self_id(self):
        return self.bot

    def get_sender_id(self):
        return self.user

    def get_group_id(self):
        return self.group or ""

    def is_private_chat(self):
        return self.private

    def get_messages(self):
        return self.parts

    def get_message_str(self):
        return self.text

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    def plain_result(self, text):
        return text


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub_patch = patch.dict(sys.modules, _modules())
        cls.stub_patch.start()
        directory = Path(__file__).resolve().parents[1]
        cls.package_name = "_directional_memory_adapter_test"
        package = ModuleType(cls.package_name)
        package.__path__ = [str(directory)]
        sys.modules[cls.package_name] = package
        spec = importlib.util.spec_from_file_location(
            cls.package_name + ".main", directory / "main.py",
        )
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        for key in list(sys.modules):
            if key == cls.package_name or key.startswith(cls.package_name + "."):
                sys.modules.pop(key, None)
        cls.stub_patch.stop()

    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        StarTools.directory = Path(self.temporary.name)
        self.config = {"max_inject_chars": 3000, "recall_limit": 20}
        self.plugin = self.module.DirectionalMemoryPlugin(SimpleNamespace(), self.config)

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.temporary.cleanup()

    async def command(self, event, action="", text=""):
        return "\n".join(
            [result async for result in self.plugin.memory_command(event, action, text)]
        )

    async def drain_captures(self):
        if self.plugin._tasks:
            await asyncio.gather(*tuple(self.plugin._tasks))
        await asyncio.sleep(0)  # Let the task completion callbacks remove tracking.

    async def test_passive_filter_captures_without_waking_or_matching_handler(self):
        event = Event("PASSIVE_SENTINEL 偏好绿茶", group="1001")
        passive = self.module.PassiveCaptureFilter()
        self.assertIs(passive.filter(event, {}), False)
        self.assertFalse(event.is_wake)
        self.assertFalse(event.is_at_or_wake_command)
        await self.drain_captures()
        self.assertFalse(event.is_wake)
        self.assertFalse(event.is_at_or_wake_command)
        reply = await self.command(Event("/dmem recall", group="1001"))
        self.assertIn("PASSIVE_SENTINEL", reply)

    async def _assert_session_gate_blocks_capture_and_injection(self, manager, method, **mock_options):
        await self.plugin.capture_message(Event("EXISTING_ALLOWED_SENTINEL"))
        event = Event("DENIED_CAPTURE_SENTINEL")
        request = self.request([TextPart(self.module.MARKER + " stale memory")])
        with patch.object(manager, method, new=AsyncMock(**mock_options)) as gate:
            self.assertIs(self.module.PassiveCaptureFilter().filter(event, {}), False)
            await self.drain_captures()
            await self.plugin.inject_memory(Event("偏好"), request)
            self.assertTrue(gate.await_count)
        self.assertEqual(request.extra_user_content_parts, [])
        reply = await self.command(Event("/dmem recall"))
        self.assertNotIn("DENIED_CAPTURE_SENTINEL", reply)
        self.assertIn("EXISTING_ALLOWED_SENTINEL", reply)

    async def test_disabled_session_blocks_passive_capture_and_injection(self):
        await self._assert_session_gate_blocks_capture_and_injection(
            self.module.SessionServiceManager, "is_session_enabled", return_value=False,
        )

    async def test_disabled_plugin_for_session_blocks_passive_capture_and_injection(self):
        await self._assert_session_gate_blocks_capture_and_injection(
            self.module.SessionPluginManager, "is_plugin_enabled_for_session", return_value=False,
        )

    async def test_session_lookup_error_fails_closed_for_capture_and_injection(self):
        await self._assert_session_gate_blocks_capture_and_injection(
            self.module.SessionServiceManager, "is_session_enabled", side_effect=RuntimeError("unavailable"),
        )

    async def test_global_whitelist_rejects_unlisted_group_before_scheduling(self):
        cfg = {"platform_settings": {"enable_id_white_list": True, "id_whitelist": ["1002"]}}
        event = Event("UNLISTED_SENTINEL", group="1001")
        self.assertIs(self.module.PassiveCaptureFilter().filter(event, cfg), False)
        self.assertFalse(self.plugin._tasks)
        self.assertFalse(event.get_extra("_directional_capture_scheduled", False))
        await self.drain_captures()
        self.assertNotIn("UNLISTED_SENTINEL", await self.command(Event("/dmem recall")))

    async def test_global_whitelist_allows_current_group_or_exact_session(self):
        for group, allowed in (("1001", ["1001"]), ("1002", ["untrusted-origin-not-used-for-identity"])):
            with self.subTest(group=group):
                event = Event("LISTED_SENTINEL_" + group, group=group)
                cfg = {"platform_settings": {"enable_id_white_list": True, "id_whitelist": allowed}}
                self.module.PassiveCaptureFilter().filter(event, cfg)
                await self.drain_captures()
                self.assertIn("LISTED_SENTINEL_" + group, await self.command(Event("/dmem recall", group=group)))

    async def test_global_plugin_set_exclusion_blocks_passive_capture(self):
        for plugin_set in ([], ["another_plugin"]):
            with self.subTest(plugin_set=plugin_set):
                event = Event("EXCLUDED_PLUGIN_SENTINEL")
                self.module.PassiveCaptureFilter().filter(event, {"plugin_set": plugin_set})
                self.assertFalse(self.plugin._tasks)
                self.assertFalse(event.get_extra("_directional_capture_scheduled", False))
        self.assertNotIn("EXCLUDED_PLUGIN_SENTINEL", await self.command(Event("/dmem recall")))

    async def test_repeated_passive_filter_schedules_same_event_only_once(self):
        event = Event("SINGLE_SCHEDULE_SENTINEL")
        passive = self.module.PassiveCaptureFilter()
        self.assertIs(passive.filter(event, {}), False)
        self.assertIs(passive.filter(event, {}), False)
        self.assertEqual(len(self.plugin._tasks), 1)
        await self.drain_captures()
        self.assertFalse(self.plugin._tasks)

    async def test_passive_capture_freezes_original_source_and_identity_before_event_rewrite(self):
        event = Event("ORIGINAL_INGRESS_SENTINEL 偏好绿茶", group="1001")
        self.module.PassiveCaptureFilter().filter(event, {})
        event.parts = [Plain("REWRITTEN_EVENT_SENTINEL")]
        event.text = "REWRITTEN_EVENT_SENTINEL"
        event.group = "1002"
        event.user = "222"
        await self.drain_captures()
        original = await self.command(Event("/dmem recall", group="1001"))
        changed = await self.command(Event("/dmem recall", group="1002", user="222"))
        self.assertIn("ORIGINAL_INGRESS_SENTINEL", original)
        self.assertNotIn("REWRITTEN_EVENT_SENTINEL", original)
        self.assertNotIn("ORIGINAL_INGRESS_SENTINEL", changed)
        self.assertNotIn("REWRITTEN_EVENT_SENTINEL", changed)

    async def test_terminate_waits_for_pending_capture_then_cleans_tasks(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def waiting_session_check(_session):
            started.set()
            await release.wait()
            return True

        with patch.object(self.module.SessionServiceManager, "is_session_enabled", new=waiting_session_check):
            self.module.PassiveCaptureFilter().filter(Event("PENDING_SENTINEL"), {})
            await asyncio.wait_for(started.wait(), timeout=1)
            termination = asyncio.create_task(self.plugin.terminate())
            await asyncio.sleep(0)
            self.assertFalse(termination.done(), "Unload must wait for the outstanding capture gate")
            release.set()
            await asyncio.wait_for(termination, timeout=1)
        self.assertFalse(self.plugin._tasks)
        self.assertTrue(self.plugin._terminating)
        self.assertIsNone(self.module._active_plugin)
        scope = self.plugin._scope(Event())
        with self.assertRaises(RuntimeError):
            self.plugin.store.search(scope)
        self.assertIs(self.module.PassiveCaptureFilter().filter(Event("AFTER_UNLOAD_SENTINEL"), {}), False)
        self.assertFalse(self.plugin._tasks)

    async def seed(self):
        for text, kwargs in (
            ("PRIVATE_ALICE 偏好秘密茶", {}),
            ("GROUP_A_ALICE 偏好甜咖啡", {"group": "1001"}),
            ("GROUP_B_ALICE 偏好冰咖啡", {"group": "1002"}),
            ("GROUP_A_BOB 偏好柠檬水", {"group": "1001", "user": "222"}),
            ("OTHER_BOT_ALICE 偏好热茶", {"bot": "998"}),
            ("OTHER_PLATFORM_ALICE 偏好果汁", {"platform": "telegram"}),
        ):
            await self.plugin.capture_message(Event(text, **kwargs))

    def request(self, parts=None):
        return SimpleNamespace(
            prompt="偏好", system_prompt="existing persona", contexts=[],
            extra_user_content_parts=list(parts or []),
        )

    def test_missing_or_invalid_identity_fails_closed(self):
        for field, value in (
            ("user", ""), ("user", None), ("user", True), ("user", "unknown"),
            ("bot", ""), ("bot", 0), ("platform", ""),
            ("platform", "none"), ("user", "bad\x00id"),
        ):
            with self.subTest(field=field, value=value):
                self.assertIsNone(self.plugin._scope(Event(**{field: value})))

    def test_group_without_trusted_group_id_is_not_private(self):
        self.assertIsNone(self.plugin._scope(Event(private=False, group="")))
        event = Event(private=False, group="1001")
        event.get_group_id = lambda: (_ for _ in ()).throw(ValueError("missing"))
        self.assertIsNone(self.plugin._scope(event))

    def test_nonboolean_chat_kind_and_self_sender_fail_closed(self):
        self.assertIsNone(self.plugin._scope(Event(private="false", group="1001")))
        self.assertIsNone(self.plugin._scope(Event(user="999", bot="999")))

    def test_allowlist_restricts_users_and_malformed_allowlist_denies(self):
        self.config["allowed_user_ids"] = ["111"]
        self.assertIsNotNone(self.plugin._scope(Event(user="111")))
        self.assertIsNone(self.plugin._scope(Event(user="222")))
        self.config["allowed_user_ids"] = "111"
        self.assertIsNone(self.plugin._scope(Event(user="111")))

    async def test_private_recall_aggregates_only_same_user_and_namespace(self):
        await self.seed()
        reply = await self.command(Event("/dmem recall 偏好"))
        for marker in ("PRIVATE_ALICE", "GROUP_A_ALICE", "GROUP_B_ALICE"):
            self.assertIn(marker, reply)
        for marker in ("GROUP_A_BOB", "OTHER_BOT_ALICE", "OTHER_PLATFORM_ALICE"):
            self.assertNotIn(marker, reply)

    async def test_group_recall_is_current_group_and_sender_only(self):
        await self.seed()
        reply = await self.command(Event("/dmem recall 偏好", group="1001"))
        self.assertIn("GROUP_A_ALICE", reply)
        for marker in ("PRIVATE_ALICE", "GROUP_B_ALICE", "GROUP_A_BOB"):
            self.assertNotIn(marker, reply)

    async def test_commands_cannot_choose_someone_else_by_putting_id_in_text(self):
        await self.seed()
        reply = await self.command(Event("/dmem recall 222 GROUP_A_BOB", group="1001"))
        self.assertNotIn("GROUP_A_BOB", reply)
        await self.command(Event("/dmem remember 222 FAKE_BOB_FACT", group="1001"))
        own = await self.command(Event("/dmem recall FAKE_BOB_FACT", group="1001"))
        bob = await self.command(Event("/dmem recall FAKE_BOB_FACT", group="1001", user="222"))
        self.assertIn("FAKE_BOB_FACT", own)
        self.assertNotIn("FAKE_BOB_FACT", bob)

    async def test_capture_skips_commands_self_and_nontext_components(self):
        for text in ("/SECRET_COMMAND", "!SECRET_COMMAND", "#SECRET_COMMAND", "dmem SECRET_COMMAND"):
            await self.plugin.capture_message(Event(text))
        await self.plugin.capture_message(Event("SELF_SENTINEL", user="999"))
        await self.plugin.capture_message(Event("FORWARD_SENTINEL", parts=[SimpleNamespace(text="FORWARD_SENTINEL")]))
        reply = await self.command(Event("/dmem recall"))
        for marker in ("SECRET_COMMAND", "SELF_SENTINEL", "FORWARD_SENTINEL"):
            self.assertNotIn(marker, reply)

    async def test_only_plain_source_text_is_captured_not_injected_prompt(self):
        event = Event("INJECTED_SENTINEL", parts=[Plain("REAL_SOURCE_SENTINEL 偏好绿茶")])
        await self.plugin.capture_message(event)
        reply = await self.command(Event("/dmem recall"))
        self.assertIn("REAL_SOURCE_SENTINEL", reply)
        self.assertNotIn("INJECTED_SENTINEL", reply)

    async def test_capture_switches_disable_only_their_source(self):
        self.config["capture_private_messages"] = False
        await self.plugin.capture_message(Event("PRIVATE_DISABLED_SENTINEL"))
        await self.plugin.capture_message(Event("GROUP_ENABLED_SENTINEL", group="1001"))
        reply = await self.command(Event("/dmem recall"))
        self.assertNotIn("PRIVATE_DISABLED_SENTINEL", reply)
        self.assertIn("GROUP_ENABLED_SENTINEL", reply)

    async def test_sensitive_capture_and_manual_memory_are_rejected(self):
        await self.plugin.capture_message(Event("password=SECRET_SENTINEL"))
        await self.command(Event("/dmem remember api_key=ANOTHER_SECRET_SENTINEL"))
        reply = await self.command(Event("/dmem recall"))
        self.assertNotIn("SECRET_SENTINEL", reply)

    async def test_synthetic_chat_plus_request_is_not_recorded(self):
        event = Event(
            "SYNTHETIC_SENTINEL", extras={
                "_group_chat_plus_request": True,
                "_group_chat_plus_current_message": "",
            },
        )
        await self.plugin.capture_message(event)
        reply = await self.command(Event("/dmem recall"))
        self.assertNotIn("SYNTHETIC_SENTINEL", reply)

    async def test_prompt_injection_is_temporary_idempotent_and_preserves_other_parts(self):
        await self.seed()
        foreign = TextPart("another plugin's content")
        request = self.request([foreign])
        await self.plugin.inject_memory(Event("偏好"), request)
        await self.plugin.inject_memory(Event("偏好"), request)
        own = [p for p in request.extra_user_content_parts if p.text.startswith(self.module.MARKER)]
        self.assertEqual(len(own), 1)
        self.assertTrue(own[0].is_temp)
        self.assertIn(foreign, request.extra_user_content_parts)
        self.assertEqual(request.prompt, "偏好")
        self.assertEqual(request.system_prompt, "existing persona")
        self.assertEqual(request.contexts, [])

    async def test_group_injected_block_never_contains_private_or_other_group_records(self):
        await self.seed()
        request = self.request()
        await self.plugin.inject_memory(Event("偏好", group="1001"), request)
        text = "\n".join(p.text for p in request.extra_user_content_parts)
        self.assertIn("GROUP_A_ALICE", text)
        for marker in ("PRIVATE_ALICE", "GROUP_B_ALICE", "GROUP_A_BOB"):
            self.assertNotIn(marker, text)

    async def _assert_long_memory_fits_serialized_budget(self, budget):
        source = 'LONG_SENTINEL 他说："喜欢\\绿茶"\n下一行\t偏好。' * 100
        await self.plugin.capture_message(Event(source))
        request = self.request()
        await self.plugin.inject_memory(Event("偏好"), request)
        blocks = [
            part.text for part in request.extra_user_content_parts
            if part.text.startswith(self.module.MARKER)
        ]
        self.assertEqual(len(blocks), 1, "Long memories must not be discarded as oversized JSON")
        block = blocks[0]
        self.assertLessEqual(len(block), budget, "Budget includes header, JSON escaping, metadata and footer")
        records = [json.loads(line) for line in block.splitlines() if line.startswith("{")]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertTrue(record["text"], "Keep a useful, nonempty portion of the remembered statement")
        self.assertIn("LONG_SENTINEL", record["text"])
        self.assertTrue(source.startswith(record["text"]))
        self.assertLess(len(record["text"]), 1200)
        self.assertIs(record.get("truncated"), True, "Partial statements must be identified explicitly")

    async def test_long_escaped_memory_is_recalled_with_minimum_total_budget(self):
        self.config["max_inject_chars"] = 300
        await self._assert_long_memory_fits_serialized_budget(300)

    async def test_long_escaped_memory_is_recalled_with_default_total_budget(self):
        self.config.pop("max_inject_chars", None)
        await self._assert_long_memory_fits_serialized_budget(1200)

    async def test_complete_memory_has_no_truncation_marker(self):
        await self.plugin.capture_message(Event("SHORT_SENTINEL 偏好绿茶"))
        request = self.request()
        await self.plugin.inject_memory(Event("偏好"), request)
        records = [
            json.loads(line)
            for part in request.extra_user_content_parts
            for line in part.text.splitlines()
            if line.startswith("{")
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text"], "SHORT_SENTINEL 偏好绿茶")
        self.assertNotIn("truncated", records[0])

    async def test_reused_request_removes_previous_memory_when_identity_becomes_invalid(self):
        await self.seed()
        request = self.request()
        await self.plugin.inject_memory(Event("偏好"), request)
        self.assertTrue(request.extra_user_content_parts)
        await self.plugin.inject_memory(Event("偏好", user=""), request)
        self.assertEqual(request.extra_user_content_parts, [])

    async def test_recall_hook_never_writes_its_injected_memory_back_to_store(self):
        await self.plugin.capture_message(Event("ORIGINAL_SENTINEL 偏好红茶", group="1001"))
        request = self.request()
        for _ in range(3):
            await self.plugin.inject_memory(Event("偏好"), request)
        reply = await self.command(Event("/dmem recall"))
        self.assertEqual(reply.count("ORIGINAL_SENTINEL"), 1)

    async def test_forget_requires_exact_confirmation(self):
        await self.seed()
        for command in ("/dmem forget", "/dmem forget confirm", "/dmem forget CONFIRM 222"):
            await self.command(Event(command, group="1001"))
        reply = await self.command(Event("/dmem recall", group="1001"))
        self.assertIn("GROUP_A_ALICE", reply)

    async def test_normalized_wake_prefix_command_preserves_full_remember_body(self):
        body = "我 喜欢 简短 回答"
        event = Event(
            "dmem remember " + body,
            parts=[Plain("bot:dmem remember " + body)],
        )
        response = await self.command(event, action="remember", text="我")
        self.assertIn("已保存", response)
        reply = await self.command(Event("/dmem recall"))
        self.assertIn(body, reply)

    async def test_forget_extra_arguments_cannot_be_truncated_to_valid_confirmation(self):
        await self.seed()
        cases = (
            ("/dmem forget CONFIRM extra", "/dmem forget CONFIRM extra"),
            ("bot:dmem forget CONFIRM extra", "dmem forget CONFIRM extra"),
            ("bot:hello", "hello"),
        )
        for raw, normalized in cases:
            with self.subTest(raw=raw):
                event = Event(normalized, group="1001", parts=[Plain(raw)])
                await self.command(event, action="forget", text="CONFIRM")
                reply = await self.command(Event("/dmem recall", group="1001"))
                self.assertIn("GROUP_A_ALICE", reply)

    async def test_group_forget_leaves_private_other_group_and_other_user_intact(self):
        await self.seed()
        await self.command(Event("/dmem forget CONFIRM", group="1001"))
        reply = await self.command(Event("/dmem recall"))
        self.assertNotIn("GROUP_A_ALICE", reply)
        self.assertIn("PRIVATE_ALICE", reply)
        self.assertIn("GROUP_B_ALICE", reply)
        bob = await self.command(Event("/dmem recall", user="222", group="1001"))
        self.assertIn("GROUP_A_BOB", bob)

    async def test_private_forget_clears_only_current_user_in_current_bot_platform(self):
        await self.seed()
        await self.command(Event("/dmem forget CONFIRM"))
        own = await self.command(Event("/dmem recall"))
        for marker in ("PRIVATE_ALICE", "GROUP_A_ALICE", "GROUP_B_ALICE"):
            self.assertNotIn(marker, own)
        bob = await self.command(Event("/dmem recall", user="222", group="1001"))
        other_bot = await self.command(Event("/dmem recall", bot="998"))
        self.assertIn("GROUP_A_BOB", bob)
        self.assertIn("OTHER_BOT_ALICE", other_bot)

    async def test_status_does_not_expose_global_counts_or_other_sources(self):
        await self.seed()
        reply = await self.command(Event("/dmem status", group="1001"))
        for marker in ("PRIVATE_ALICE", "GROUP_B_ALICE", "GROUP_A_BOB", "1002", "222"):
            self.assertNotIn(marker, reply)

    async def test_disabled_or_disallowed_user_has_no_capture_or_recall(self):
        self.config["allowed_user_ids"] = ["222"]
        await self.plugin.capture_message(Event("DISALLOWED_SENTINEL"))
        result = await self.command(Event("/dmem remember DISALLOWED_SENTINEL"))
        self.assertNotIn("已保存", result)
        self.config["allowed_user_ids"] = []
        self.assertNotIn("DISALLOWED_SENTINEL", await self.command(Event("/dmem recall")))
        self.config["enabled"] = False
        await self.plugin.capture_message(Event("DISABLED_SENTINEL"))
        self.config["enabled"] = True
        self.assertNotIn("DISABLED_SENTINEL", await self.command(Event("/dmem recall")))


if __name__ == "__main__":
    unittest.main()
