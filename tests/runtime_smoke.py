"""Smoke-test the adapter against a real AstrBot runtime, without sending messages.

Run separately from unittest discovery inside an AstrBot 4.29.3 environment::

    PYTHONPATH=/AstrBot python -B /work/tests/runtime_smoke.py

Only synthetic events and a temporary SQLite database are used. Session permission
lookups are mocked so the running bot's session configuration is never consulted.
An installation without AstrBot prints the required environment and exits with 2.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
import inspect
import itertools
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch


def load_plugin():
    """Import the real adapter and its real decorators in an isolated namespace."""
    directory = Path(__file__).resolve().parents[1]
    package_name = "_synthetic_directional_memory_runtime_smoke"
    package = ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        package_name + ".main", directory / "main.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the directional-memory adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def smoke() -> int:
    from astrbot.api import logger
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.message_components import Plain
    from astrbot.api.platform import PlatformMetadata
    from astrbot.api.star import StarTools
    from astrbot.core.agent.message import TextPart
    from astrbot.core.platform.astrbot_message import AstrBotMessage
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.provider.entities import ProviderRequest
    from astrbot.core.star.filter.command import CommandFilter

    # This script is its own process; keep successful output to one pass count.
    logging.disable(logging.CRITICAL)
    if callable(getattr(logger, "remove", None)):
        logger.remove()
    module = load_plugin()
    serial = itertools.count()
    passed = 0

    def check(condition, description):
        nonlocal passed
        if not condition:
            raise AssertionError(description)
        passed += 1

    def event(
        text="偏好", *, user="synthetic-alice", group=None,
        bot="synthetic-bot", platform="aiocqhttp", group_message=None,
    ):
        is_group = group is not None if group_message is None else group_message
        message = AstrBotMessage()
        message.type = MessageType.GROUP_MESSAGE if is_group else MessageType.FRIEND_MESSAGE
        message.self_id = bot
        message.sender = SimpleNamespace(user_id=user, nickname="Synthetic test user")
        message.group_id = group
        message.session_id = group if is_group else user
        message.message = [Plain(text=text)]
        message.message_str = text
        message.message_id = "synthetic-event-" + str(next(serial))
        message.raw_message = {}
        return AstrMessageEvent(
            text, message,
            PlatformMetadata(
                name=platform, description="Synthetic smoke test",
                id="synthetic-platform-" + platform,
            ),
            session_id=str(group if is_group else user),
        )

    seeds = (
        ("SYNTHETIC_ALICE_PRIVATE 偏好绿茶", {}),
        ("SYNTHETIC_ALICE_GROUP_A 偏好甜咖啡", {"group": "synthetic-group-a"}),
        ("SYNTHETIC_ALICE_GROUP_B 偏好冰咖啡", {"group": "synthetic-group-b"}),
        ("SYNTHETIC_BOB_GROUP_A 偏好柠檬水", {"user": "synthetic-bob", "group": "synthetic-group-a"}),
        ("SYNTHETIC_BOB_PRIVATE 偏好红茶", {"user": "synthetic-bob"}),
        ("SYNTHETIC_OTHER_BOT 偏好热茶", {"bot": "synthetic-other-bot"}),
        ("SYNTHETIC_OTHER_PLATFORM 偏好果汁", {"platform": "synthetic-other-platform"}),
    )
    markers = {text.split()[0] for text, _ in seeds}
    matrix = (
        ({}, {"SYNTHETIC_ALICE_PRIVATE", "SYNTHETIC_ALICE_GROUP_A", "SYNTHETIC_ALICE_GROUP_B"}),
        ({"group": "synthetic-group-a"}, {"SYNTHETIC_ALICE_GROUP_A"}),
        ({"group": "synthetic-group-b"}, {"SYNTHETIC_ALICE_GROUP_B"}),
        ({"user": "synthetic-bob"}, {"SYNTHETIC_BOB_PRIVATE", "SYNTHETIC_BOB_GROUP_A"}),
        ({"user": "synthetic-bob", "group": "synthetic-group-a"}, {"SYNTHETIC_BOB_GROUP_A"}),
        ({"bot": "synthetic-other-bot"}, {"SYNTHETIC_OTHER_BOT"}),
        ({"platform": "synthetic-other-platform"}, {"SYNTHETIC_OTHER_PLATFORM"}),
    )
    config = {"enabled": True, "recall_limit": 20, "max_inject_chars": 6000}
    with tempfile.TemporaryDirectory(prefix="directional-memory-synthetic-") as directory:
        with (
            patch.object(StarTools, "get_data_dir", return_value=Path(directory)),
            patch.object(module.SessionServiceManager, "is_session_enabled", new=AsyncMock(return_value=True)),
            patch.object(module.SessionPluginManager, "is_plugin_enabled_for_session", new=AsyncMock(return_value=True)),
        ):
            plugin = module.DirectionalMemoryPlugin(None, config)
            try:
                for text, namespace in seeds:
                    await plugin.capture_message(event(text, **namespace))

                for namespace, expected in matrix:
                    foreign = TextPart(text="Synthetic content from another plugin")
                    request = ProviderRequest(
                        prompt="偏好", system_prompt="Synthetic existing persona",
                        contexts=[{"role": "user", "content": "Synthetic context"}],
                        extra_user_content_parts=[foreign],
                    )
                    original_contexts = request.contexts
                    context_snapshot = deepcopy(request.contexts)
                    await plugin.inject_memory(event(**namespace), request)
                    await plugin.inject_memory(event(**namespace), request)
                    own = [
                        part for part in request.extra_user_content_parts
                        if isinstance(part, TextPart) and part.text.startswith(module.MARKER)
                    ]
                    check(len(own) == 1, "Repeated real ProviderRequest injection must be idempotent")
                    check(getattr(own[0], "_no_save", False) is True, "Injected TextPart must carry the real _no_save flag")
                    check(
                        request.contexts is original_contexts and request.contexts == context_snapshot,
                        "Existing conversation contexts must remain unchanged",
                    )
                    check(
                        request.prompt == "偏好" and request.system_prompt == "Synthetic existing persona"
                        and any(part is foreign for part in request.extra_user_content_parts),
                        "Existing prompt, persona and another plugin's content must survive",
                    )
                    records = [
                        json.loads(line) for line in own[0].text.splitlines()
                        if line.startswith("{")
                    ]
                    visible = {record["text"].split()[0] for record in records}
                    check(visible == expected, "Real event visibility matrix failed: " + repr(namespace))
                    check(
                        not any(marker in own[0].text for marker in markers - expected),
                        "Forbidden user, group, bot or platform content leaked into the prompt",
                    )

                check(
                    plugin._scope(event("Unknown group", group_message=True)) is None,
                    "A group event without a trusted group id must fail closed",
                )
                check(
                    plugin._scope(event("Bot echo", user="synthetic-bot")) is None,
                    "The bot's own messages must fail closed",
                )

                # Exercise AstrBot's actual command parser, including its ordinary
                # string parameters, then ensure the adapter retains all raw words.
                fact = "I prefer quiet mornings and unsweetened green tea"
                command_event = event("/dmem remember " + fact, group="synthetic-group-a")
                command_event.message_str = "dmem remember " + fact  # Framework removes its wake prefix.
                command_event.is_at_or_wake_command = True
                command_filter = CommandFilter(
                    "dmem", handler_md=SimpleNamespace(handler=module.DirectionalMemoryPlugin.memory_command),
                )
                accepted = command_filter.filter(command_event, {})
                if inspect.isawaitable(accepted):
                    accepted = await accepted
                check(bool(accepted), "The real dmem CommandFilter must accept the synthetic command")
                parameters = command_event.get_extra("parsed_params")
                check(
                    isinstance(parameters, dict) and parameters.get("action") == "remember",
                    "The real CommandFilter must supply parsed handler parameters",
                )
                responses = [
                    result async for result in plugin.memory_command(command_event, **parameters)
                ]
                check(bool(responses), "The real command must yield a response without sending it")
                own_group = plugin.store.search(plugin._scope(command_event), query="", limit=20, max_chars=6000)
                check(
                    any(record.kind == "fact" and record.text == fact for record in own_group),
                    "A multiword remember command must preserve the complete fact",
                )
                other_group = plugin.store.search(
                    plugin._scope(event(group="synthetic-group-b")), query="", limit=20, max_chars=6000,
                )
                check(not any(record.text == fact for record in other_group), "Manual facts must not cross group boundaries")
                private_records = plugin.store.search(plugin._scope(event()), query="", limit=20, max_chars=6000)
                check(any(record.text == fact for record in private_records), "Private recall must include the user's manual group fact")
            finally:
                await plugin.terminate()
    return passed


def main() -> int:
    try:
        import astrbot  # noqa: F401 - Fail clearly before importing the adapter.
    except ModuleNotFoundError as exc:
        if exc.name != "astrbot":
            raise
        print("AstrBot runtime required: run this script in the AstrBot 4.29.3 container with PYTHONPATH=/AstrBot.")
        return 2
    passed = asyncio.run(smoke())
    print(f"{passed} runtime smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
