"""
Tests for QDRANT_ALLOWED_COLLECTION_PREFIXES — the collection-namespace lock.

Covers: settings parsing (empty-element hygiene), the pure validators
(fail-closed on a degenerate armed allowlist, case-sensitivity — these live
here with the feature rather than in tests/test_validators.py; see that file
for the sibling validator suites), and the tool-layer guard driven through the
real tool closures via FastMCP.call_tool (which re-raises tool exceptions as
ToolError — asserting ValueError would miss). The listing filter test seeds a
non-admitted collection via the CONNECTOR directly (the guard lives in the
tool closure, so the connector bypasses it into the same :memory: client) —
without seeding, the filter assertion would be vacuously true on an empty
instance.

The protocol-layer shape of a rejection (result.isError=true, what an actual
MCP client sees) is CI-covered via the in-memory client session
(test_wire_shape_rejection_is_iserror); the opt-in live smoke
(scripts/roundtrip_smoke.py, SMOKE_EXPECT_PREFIX_LOCK) remains the
live-HTTP-transport belt on top of it.
"""

import uuid

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from tests.conftest import make_server as _make_server
from tests.conftest import tool_json
from tests.conftest import tool_text as _text

from mcp_server_qdrant.enhanced_qdrant import Entry
from mcp_server_qdrant.enhanced_settings import EnhancedQdrantSettings
from mcp_server_qdrant.validators import (
    filter_allowed_collections,
    is_collection_allowed,
)

# Shared harness (make_server/tool_text/tool_json + the locked_server/
# unlocked_server fixtures) lives in tests/conftest.py so sibling suites
# reuse it.


class TestSettingsParsing:
    """QDRANT_ALLOWED_COLLECTION_PREFIXES parsing on EnhancedQdrantSettings."""

    def _prefixes(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("QDRANT_ALLOWED_COLLECTION_PREFIXES", raising=False)
        else:
            monkeypatch.setenv("QDRANT_ALLOWED_COLLECTION_PREFIXES", value)
        return EnhancedQdrantSettings().get_allowed_collection_prefixes()

    def test_unset_means_disabled(self, monkeypatch):
        assert self._prefixes(monkeypatch, None) == []

    def test_set_but_empty_means_disabled(self, monkeypatch):
        assert self._prefixes(monkeypatch, "") == []

    def test_single_prefix(self, monkeypatch):
        assert self._prefixes(monkeypatch, "mld_") == ["mld_"]

    def test_multiple_prefixes(self, monkeypatch):
        assert self._prefixes(monkeypatch, "mld_,test_") == ["mld_", "test_"]

    def test_whitespace_stripped(self, monkeypatch):
        assert self._prefixes(monkeypatch, " mld_ , test_ ") == ["mld_", "test_"]

    def test_trailing_comma_never_yields_empty_element(self, monkeypatch):
        # A "" element would make startswith() allow-all — the silent-unlock trap.
        assert self._prefixes(monkeypatch, "mld_,") == ["mld_"]

    def test_only_separators_means_disabled(self, monkeypatch):
        assert self._prefixes(monkeypatch, " , ") == []
        assert self._prefixes(monkeypatch, ",,") == []


class TestPrefixValidators:
    def test_empty_allowlist_allows_all(self):
        assert is_collection_allowed("anything", []) is True

    def test_matching_prefix_allowed(self):
        assert is_collection_allowed("mld_sessions", ["mld_"]) is True

    def test_non_matching_rejected(self):
        assert is_collection_allowed("other_project", ["mld_"]) is False

    def test_empty_element_never_opens_the_gate(self):
        # Defense-in-depth: even if a parse regression admits "", the matcher
        # must not treat it as allow-all.
        assert is_collection_allowed("other_x", ["mld_", ""]) is False
        assert is_collection_allowed("mld_x", ["mld_", ""]) is True

    def test_armed_but_degenerate_allowlist_fails_closed(self):
        # An allowlist that is non-empty but contains ONLY falsy elements is
        # ARMED-but-degenerate (a parse regression) — deny, don't disarm.
        assert is_collection_allowed("anything", [""]) is False
        assert is_collection_allowed("mld_x", [""]) is False

    def test_prefix_boundary_near_miss_rejected(self):
        # startswith semantics: the underscore is part of the prefix.
        assert is_collection_allowed("mld", ["mld_"]) is False
        assert is_collection_allowed("mldx_other", ["mld_"]) is False

    def test_second_prefix_admits(self):
        # A bug that only checked prefixes[0] must not pass.
        assert is_collection_allowed("test_a", ["mld_", "test_"]) is True

    def test_prefix_matching_is_literal_not_pattern(self):
        # startswith is literal; a "smarter" regex/glob matcher would let a
        # metacharacter-bearing prefix silently widen the namespace.
        assert is_collection_allowed("mldX", ["mld."]) is False
        assert is_collection_allowed("mld.", ["mld."]) is True

    def test_empty_name_rejected_when_armed(self):
        # "" is a client-suppliable str; the mirror image of the empty-PREFIX
        # cases — an empty NAME must fail closed under an armed lock.
        assert is_collection_allowed("", ["mld_"]) is False
        assert is_collection_allowed("   ", ["mld_"]) is False

    def test_case_sensitive_by_design(self):
        # MLD_x is a DIFFERENT physical Qdrant collection than mld_x; a
        # case-insensitive "fix" would let MLD_* join the locked namespace.
        assert is_collection_allowed("MLD_x", ["mld_"]) is False

    def test_non_string_rejected(self):
        assert is_collection_allowed(None, ["mld_"]) is False
        assert is_collection_allowed(123, ["mld_"]) is False

    def test_filter_allowed_collections(self):
        assert filter_allowed_collections(["mld_a", "b", "mld_c"], ["mld_"]) == [
            "mld_a",
            "mld_c",
        ]
        assert filter_allowed_collections(["a", "b"], []) == ["a", "b"]


class TestToolGuardLocked:
    """The 7-tool entry guard, driven through the real closures via call_tool.

    call_tool re-raises tool exceptions wrapped as ToolError (NOT the
    original ValueError) — over the HTTP protocol the same rejection surfaces
    as result.isError instead.
    """

    async def test_store_rejected(self, locked_server):
        with pytest.raises(ToolError, match="not allowed on this endpoint"):
            await locked_server.call_tool(
                "qdrant_store",
                {"information": "x", "collection_name": "other_project"},
            )

    async def test_find_rejected_not_swallowed(self, locked_server):
        # The guard must sit ABOVE qdrant_find's broad try/except — a swallowed
        # rejection would return a success-shaped error dict instead of raising.
        with pytest.raises(ToolError, match="not allowed on this endpoint"):
            await locked_server.call_tool(
                "qdrant_find",
                {"query": "q", "collection_name": "other_project"},
            )

    async def test_collection_info_rejected_not_swallowed(self, locked_server):
        with pytest.raises(ToolError, match="not allowed on this endpoint"):
            await locked_server.call_tool(
                "qdrant_collection_info", {"collection_name": "other_project"}
            )

    async def test_bulk_store_rejected(self, locked_server):
        with pytest.raises(ToolError, match="not allowed on this endpoint"):
            await locked_server.call_tool(
                "qdrant_bulk_store",
                {"documents": ["x"], "collection_name": "other_project"},
            )

    async def test_get_point_rejected(self, locked_server):
        with pytest.raises(ToolError, match="not allowed on this endpoint"):
            await locked_server.call_tool(
                "qdrant_get_point",
                {"point_id": uuid.uuid4().hex, "collection_name": "other_project"},
            )

    async def test_update_payload_rejected(self, locked_server):
        with pytest.raises(ToolError, match="not allowed on this endpoint"):
            await locked_server.call_tool(
                "qdrant_update_payload",
                {
                    "point_ids": [uuid.uuid4().hex],
                    "payload": {"k": "v"},
                    "collection_name": "other_project",
                },
            )

    async def test_delete_points_rejected(self, locked_server):
        with pytest.raises(ToolError, match="not allowed on this endpoint"):
            await locked_server.call_tool(
                "qdrant_delete_points",
                {
                    "point_ids": [uuid.uuid4().hex],
                    "collection_name": "other_project",
                },
            )

    async def test_admitted_prefix_accepted(self, locked_server):
        result = await locked_server.call_tool(
            "qdrant_store",
            {"information": "hello archive", "collection_name": "mld_guard_test"},
        )
        assert "mld_guard_test" in _text(result)

    async def test_admitted_find_succeeds_under_lock(self, locked_server):
        # Locked SUCCESS path for qdrant_find: the guard must not disturb a
        # legitimate admitted-collection search (guard-then-try interaction).
        # Assert the STRUCTURED payload — a text-substring disjunction is
        # satisfiable by the tool's swallowed-error dict (it also carries
        # total_found/collection keys), which would false-green a broken
        # search.
        await locked_server.call_tool(
            "qdrant_store",
            {
                "information": "archive fact about widgets",
                "collection_name": "mld_find_test",
            },
        )
        payload = tool_json(
            await locked_server.call_tool(
                "qdrant_find",
                {"query": "widgets", "collection_name": "mld_find_test"},
            )
        )
        assert "error" not in payload, payload
        assert payload["total_found"] >= 1, payload

    async def test_rejection_message_carries_name_and_prefixes(self, locked_server):
        # The operator-actionable halves of the message (the offending name +
        # the configured allowlist) must not silently degrade.
        with pytest.raises(ToolError) as exc:
            await locked_server.call_tool(
                "qdrant_store",
                {"information": "x", "collection_name": "other_project"},
            )
        assert "other_project" in str(exc.value)
        assert "mld_" in str(exc.value)

    async def test_second_prefix_admits_through_tool(self, monkeypatch):
        # prefixes[0]-only regression pinned at the integration layer too.
        server = _make_server(monkeypatch, prefixes="mld_,test_")
        result = await server.call_tool(
            "qdrant_store",
            {"information": "x", "collection_name": "test_second_prefix"},
        )
        assert "test_second_prefix" in _text(result)

    async def test_empty_collection_name_rejected_under_lock(self, locked_server):
        # An empty string passes the str schema and reaches the guard — it
        # must fail closed, not fall through to the connector.
        with pytest.raises(ToolError, match="not allowed on this endpoint"):
            await locked_server.call_tool(
                "qdrant_store", {"information": "x", "collection_name": ""}
            )

    async def test_guard_precedes_other_arg_validation(self, locked_server):
        # The guard is contractually the FIRST statement — a foreign
        # collection must yield the prefix rejection even when another arg is
        # ALSO invalid (mismatched metadata_list). Pins the insertion point
        # future optional-arg validation (e.g. client point IDs) sits below.
        with pytest.raises(ToolError, match="not allowed on this endpoint"):
            await locked_server.call_tool(
                "qdrant_bulk_store",
                {
                    "documents": ["a", "b"],
                    "collection_name": "other_project",
                    "metadata_list": [{"k": "v"}],
                },
            )

    async def test_every_collection_taking_tool_is_guarded(self, locked_server):
        # Guard-completeness drift check: derive the collection-taking tool
        # set from the REGISTERED SCHEMAS (not a hand list), then drive each
        # through call_tool with a foreign name. A future tool that takes
        # collection_name without the guard turns this red the day it lands.
        tools = await locked_server.list_tools()
        takers = {
            t.name
            for t in tools
            if "collection_name" in (t.inputSchema or {}).get("properties", {})
        }
        MINIMAL_ARGS = {
            "qdrant_store": {"information": "x", "collection_name": "other_project"},
            "qdrant_find": {"query": "x", "collection_name": "other_project"},
            "qdrant_collection_info": {"collection_name": "other_project"},
            "qdrant_bulk_store": {
                "documents": ["x"],
                "collection_name": "other_project",
            },
            "qdrant_get_point": {
                "point_id": uuid.uuid4().hex,
                "collection_name": "other_project",
            },
            "qdrant_update_payload": {
                "point_ids": [uuid.uuid4().hex],
                "payload": {"k": "v"},
                "collection_name": "other_project",
            },
            "qdrant_delete_points": {
                "point_ids": [uuid.uuid4().hex],
                "collection_name": "other_project",
            },
        }
        assert takers == set(MINIMAL_ARGS), (
            "collection-taking tool set drifted — add the guard AND a "
            f"MINIMAL_ARGS entry for: {takers ^ set(MINIMAL_ARGS)}"
        )
        for name in sorted(takers):
            with pytest.raises(ToolError, match="not allowed on this endpoint"):
                await locked_server.call_tool(name, MINIMAL_ARGS[name])

    async def test_wire_shape_rejection_is_iserror(self, locked_server):
        # The shape an actual MCP CLIENT sees: over the protocol layer the
        # guard's raise must surface as isError=true with the message — not a
        # success-shaped result. (In-memory client session = the real
        # protocol conversion, no live server needed; the HTTP smoke remains
        # the live-transport belt.)
        from mcp.shared.memory import (
            create_connected_server_and_client_session as connect,
        )

        async with connect(locked_server._mcp_server) as session:
            r = await session.call_tool(
                "qdrant_store",
                {"information": "x", "collection_name": "other_project"},
            )
            assert r.isError is True
            assert "not allowed on this endpoint" in (
                r.content[0].text if r.content else ""
            )


class TestListingFiltersLocked:
    async def test_list_collections_excludes_foreign(self, locked_server):
        # Seed a NON-admitted collection via the connector (bypasses the tool
        # guard into the same :memory: client) — otherwise this assertion is
        # vacuously true on an empty instance.
        await locked_server.qdrant_connector.store(
            Entry(content="foreign doc"), collection_name="other_project"
        )
        await locked_server.qdrant_connector.store(
            Entry(content="ours"), collection_name="mld_guard_test"
        )
        # Non-vacuity pin: the foreign collection REALLY exists (unfiltered,
        # connector level) before we assert the tool hides it.
        names = await locked_server.qdrant_connector.get_collection_names()
        assert "other_project" in names
        text = _text(await locked_server.call_tool("qdrant_list_collections", {}))
        assert "mld_guard_test" in text
        assert "other_project" not in text

    async def test_list_collections_never_fetches_foreign_info(
        self, locked_server, monkeypatch
    ):
        # The contract's stronger half: a non-admitted name must never even be
        # FETCHED (names filter before the per-collection info loop) — not
        # merely hidden at render time.
        await locked_server.qdrant_connector.store(
            Entry(content="foreign doc"), collection_name="other_project"
        )
        await locked_server.qdrant_connector.store(
            Entry(content="ours"), collection_name="mld_guard_test"
        )
        fetched = []
        real_get_info = locked_server.qdrant_connector.get_collection_info

        async def spy(name):
            fetched.append(name)
            return await real_get_info(name)

        monkeypatch.setattr(locked_server.qdrant_connector, "get_collection_info", spy)
        await locked_server.call_tool("qdrant_list_collections", {})
        assert fetched, "spy never fired — test is vacuous"
        assert all(n.startswith("mld_") for n in fetched), fetched

    async def test_model_mappings_shows_env_mld_keys_and_no_foreign_names(
        self, monkeypatch
    ):
        server = _make_server(
            monkeypatch,
            prefixes="mld_",
            env_mappings='{"mld_sessions": "bge-large-en-v1.5"}',
        )
        text = _text(await server.call_tool("qdrant_model_mappings", {}))
        # The effective (env-merged) mld_ mapping appears...
        assert "mld_sessions" in text
        # ...and no non-admitted collection name leaks (module defaults are
        # all non-mld_): EVERY rendered mapping row must be mld_-prefixed,
        # not just the one representative name.
        import re as _re

        mapping_rows = _re.findall(
            r"\*\*(.+?)\*\*", text.split("Available Model Configs")[0]
        )
        collection_rows = [r for r in mapping_rows if "Mappings" not in r]
        assert collection_rows, "no mapping rows rendered — test is vacuous"
        assert all(r.startswith("mld_") for r in collection_rows), collection_rows
        assert "legal_analysis" not in text
        # The model-CONFIG section is deliberately NOT filtered (keys are
        # model names, no lock-relevant info) — an over-filter regression
        # that emptied it must be caught.
        assert "bge-large-en-v1.5" in text
        assert "1024" in text

    async def test_model_mappings_survives_non_dict_env_json(self, monkeypatch):
        # COLLECTION_MODEL_MAPPINGS='[1,2,3]' is valid JSON (passes the
        # settings validator, which only checks parseability) but not an
        # object — the merge must fail soft, not crash the tool. The name
        # filter must ALSO still apply on this degraded path (the overlay
        # collapses to {} and the module defaults must still be dropped).
        server = _make_server(monkeypatch, prefixes="mld_", env_mappings="[1, 2, 3]")
        text = _text(await server.call_tool("qdrant_model_mappings", {}))
        assert "Available Model Configs" in text  # rendered cleanly
        assert "legal_analysis" not in text  # filter survived the fail-soft

    @pytest.mark.parametrize(
        "env_mappings,env_configs",
        [
            (
                '{"mld_x": [1, 2]}',
                None,
            ),  # unhashable mapping VALUE → .get raises TypeError
            (
                None,
                '{"m": "notadict"}',
            ),  # non-dict config VALUE → .get raises AttributeError
        ],
    )
    async def test_model_mappings_survives_malformed_env_values(
        self, monkeypatch, env_mappings, env_configs
    ):
        # Values inside a parseable JSON object are NOT validated upstream —
        # an operator typo must degrade the dump, never crash the tool.
        server = _make_server(
            monkeypatch,
            prefixes="mld_",
            env_mappings=env_mappings,
            env_configs=env_configs,
        )
        text = _text(await server.call_tool("qdrant_model_mappings", {}))
        assert "Available Model Configs" in text

    async def test_model_mappings_custom_configs_overlay_renders(self, monkeypatch):
        # The second _merged call site (CUSTOM_MODEL_CONFIGS) — an env-supplied
        # custom config must actually land in the locked dump; a wiring
        # regression (wrong raw string passed) stays green without this.
        server = _make_server(
            monkeypatch,
            prefixes="mld_",
            env_mappings='{"mld_sessions": "my-model"}',
            env_configs='{"my-model": {"dimensions": 512, "fastembed_model": "x/y"}}',
        )
        text = _text(await server.call_tool("qdrant_model_mappings", {}))
        assert "my-model" in text
        assert "512" in text

    async def test_model_mappings_locked_default_env_is_empty_but_clean(
        self, locked_server
    ):
        # The common locked-endpoint shape (no env mappings): every module
        # default is non-admitted → empty mappings section, but the configs
        # section must still render and no foreign name may appear.
        text = _text(await locked_server.call_tool("qdrant_model_mappings", {}))
        assert "legal_analysis" not in text
        assert "Available Model Configs" in text
        assert "bge-large-en-v1.5" in text

    async def test_list_collections_all_filtered_renders_empty_message(
        self, locked_server
    ):
        # Locked endpoint over an instance holding ONLY foreign collections
        # (the accidental-bleed shape): everything filters out → the
        # no-collections branch must render, leaking nothing.
        await locked_server.qdrant_connector.store(
            Entry(content="foreign doc"), collection_name="other_project"
        )
        text = _text(await locked_server.call_tool("qdrant_list_collections", {}))
        assert "No collections found" in text
        assert "other_project" not in text


class TestUnlockedBackwardCompat:
    """Env absent ⇒ exactly today's behavior (the shared-instance contract)."""

    async def test_store_any_collection(self, unlocked_server):
        result = await unlocked_server.call_tool(
            "qdrant_store",
            {"information": "hello", "collection_name": "any_project_x"},
        )
        assert "any_project_x" in _text(result)

    async def test_list_collections_unfiltered(self, unlocked_server):
        await unlocked_server.qdrant_connector.store(
            Entry(content="doc"), collection_name="any_project_x"
        )
        text = _text(await unlocked_server.call_tool("qdrant_list_collections", {}))
        assert "any_project_x" in text

    async def test_model_mappings_shows_defaults(self, unlocked_server):
        text = _text(await unlocked_server.call_tool("qdrant_model_mappings", {}))
        assert "legal_analysis" in text

    async def test_model_mappings_unlocked_ignores_env_mappings(self, monkeypatch):
        # Flag-off no-op contract: with the lock UNSET, the dump stays the
        # module-constant output even when env mappings are configured — the
        # env-merge is a locked-endpoint behavior only.
        server = _make_server(
            monkeypatch,
            prefixes=None,
            env_mappings='{"mld_sessions": "bge-large-en-v1.5"}',
        )
        text = _text(await server.call_tool("qdrant_model_mappings", {}))
        assert "mld_sessions" not in text
        assert "legal_analysis" in text

    async def test_no_tool_rejects_when_unlocked(self, unlocked_server):
        # None of the 7 guarded tools may emit the namespace rejection when
        # the lock is unset (other errors — e.g. collection-not-found shapes —
        # are fine; only the "not allowed on this endpoint" text is forbidden).
        pid = uuid.uuid4().hex
        calls = [
            ("qdrant_store", {"information": "x", "collection_name": "any_c"}),
            ("qdrant_find", {"query": "x", "collection_name": "any_c"}),
            ("qdrant_collection_info", {"collection_name": "any_c"}),
            ("qdrant_bulk_store", {"documents": ["x"], "collection_name": "any_c"}),
            ("qdrant_get_point", {"point_id": pid, "collection_name": "any_c"}),
            (
                "qdrant_update_payload",
                {"point_ids": [pid], "payload": {"k": "v"}, "collection_name": "any_c"},
            ),
            ("qdrant_delete_points", {"point_ids": [pid], "collection_name": "any_c"}),
        ]
        for tool, args in calls:
            try:
                result = await unlocked_server.call_tool(tool, args)
                assert "not allowed on this endpoint" not in _text(result), tool
            except ToolError as e:
                assert "not allowed on this endpoint" not in str(e), tool


class TestStartupLockStateLog:
    """The launch-visible safety signal (armed vs DISABLED) must not silently
    disappear — it exists so a misconfigured unlocked dedicated endpoint is
    seen at startup, not discovered by the absence of rejections."""

    def test_logs_armed_state(self, monkeypatch, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="mcp_server_qdrant.mcp_server"):
            _make_server(monkeypatch, prefixes="mld_,test_")
        assert any(
            "collection prefix lock: mld_,test_" in r.getMessage()
            for r in caplog.records
        )

    def test_logs_disabled_state(self, monkeypatch, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="mcp_server_qdrant.mcp_server"):
            _make_server(monkeypatch, prefixes=None)
        assert any(
            "collection prefix lock: DISABLED" in r.getMessage() for r in caplog.records
        )

    def test_separators_only_env_logs_disabled(self, monkeypatch, caplog):
        # The exact silent-unlock trap × startup-visibility composition: a
        # value that LOOKS armed in the shell (" , " / ",,") parses to [] —
        # the log must say DISABLED, and must never echo the raw separators
        # as if armed.
        import logging

        for raw in (" , ", ",,"):
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="mcp_server_qdrant.mcp_server"):
                _make_server(monkeypatch, prefixes=raw)
            lock_lines = [
                r.getMessage()
                for r in caplog.records
                if "collection prefix lock" in r.getMessage()
            ]
            assert lock_lines, f"no lock-state line for {raw!r}"
            assert all("DISABLED" in ln for ln in lock_lines), lock_lines
