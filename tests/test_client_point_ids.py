"""
Tests for client-supplied point IDs (qdrant_store point_id /
qdrant_bulk_store point_ids) — deterministic addressing for idempotent
upsert + read-back-confirm workflows.

The ID-form contract is strict-accept: ONLY the canonical lowercase
hyphenated UUID string (str(uuid.UUID(...))) is admitted. Qdrant's local
:memory: mode keys points by the RAW string while the real server keys by
the parsed UUID value — the canonical form is the only spelling that is
byte-identical on both backends, and rejecting the rest loudly means a
caller can never store an ID it cannot read back.
"""

import uuid

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from tests.conftest import tool_json

from mcp_server_qdrant.enhanced_qdrant import Entry
from mcp_server_qdrant.validators import is_valid_point_id

NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # any fixed namespace


def canonical(key: str) -> str:
    return str(uuid.uuid5(NS, key))


class TestPointIdValidator:
    def test_canonical_hyphenated_accepted(self):
        assert is_valid_point_id(str(uuid.uuid4())) is True
        assert is_valid_point_id(canonical("source:1")) is True

    def test_hex_form_rejected(self):
        # Today's SERVER-minted form — but a client-supplied hex string is
        # not byte-identical to its canonical spelling, so local-mode
        # read-back by the canonical form would miss it. Reject loudly.
        assert is_valid_point_id(uuid.uuid4().hex) is False

    def test_urn_and_braced_and_upper_rejected(self):
        u = uuid.uuid4()
        assert is_valid_point_id(f"urn:uuid:{u}") is False
        assert is_valid_point_id("{" + str(u) + "}") is False
        assert is_valid_point_id(str(u).upper()) is False

    def test_non_uuid_and_non_string_rejected(self):
        assert is_valid_point_id("not-a-uuid") is False
        assert is_valid_point_id("") is False
        assert is_valid_point_id(None) is False
        assert is_valid_point_id(123) is False
        assert is_valid_point_id(uuid.uuid4()) is False  # UUID object, not str


class TestStoreWithPointId:
    async def test_store_then_get_point_round_trip(self, unlocked_server):
        pid = canonical("memory:file-a")
        await unlocked_server.call_tool(
            "qdrant_store",
            {
                "information": "evicted memory body",
                "collection_name": "pid_rt_test",
                "point_id": pid,
            },
        )
        payload = tool_json(
            await unlocked_server.call_tool(
                "qdrant_get_point",
                {"point_id": pid, "collection_name": "pid_rt_test"},
            )
        )
        assert "error" not in payload, payload
        assert payload["id"] == pid
        assert "evicted memory body" in str(payload.get("payload", {}))

    async def test_restore_same_id_overwrites_not_duplicates(self, unlocked_server):
        # §6 upsert idempotency: a retried/re-run write cannot duplicate.
        pid = canonical("memory:file-b")
        for content in ("first version", "second version"):
            await unlocked_server.call_tool(
                "qdrant_store",
                {
                    "information": content,
                    "collection_name": "pid_upsert_test",
                    "point_id": pid,
                },
            )
        info = await unlocked_server.qdrant_connector.get_collection_info(
            "pid_upsert_test"
        )
        assert info.get("points_count") == 1, info
        payload = tool_json(
            await unlocked_server.call_tool(
                "qdrant_get_point",
                {"point_id": pid, "collection_name": "pid_upsert_test"},
            )
        )
        assert "second version" in str(payload.get("payload", {}))

    async def test_omitted_point_id_mints_server_side(self, unlocked_server):
        # Backward compat: the shared instance never sets the param — the
        # server-minted uuid4 hex path must behave exactly as today.
        await unlocked_server.call_tool(
            "qdrant_store",
            {"information": "no explicit id", "collection_name": "pid_default_test"},
        )
        payload = tool_json(
            await unlocked_server.call_tool(
                "qdrant_find",
                {"query": "no explicit id", "collection_name": "pid_default_test"},
            )
        )
        assert payload["total_found"] >= 1
        minted = payload["results"][0]["point_id"]
        # :memory:-specific assertion: local mode preserves the raw stored
        # string (hex stays hex); a REAL server would return the canonical
        # hyphenated rendering. This suite always runs on :memory:.
        assert len(minted) == 32 and "-" not in minted  # uuid4().hex form
        uuid.UUID(minted)  # parses

    async def test_non_canonical_form_rejected(self, unlocked_server):
        for bad in (
            uuid.uuid4().hex,
            f"urn:uuid:{uuid.uuid4()}",
            str(uuid.uuid4()).upper(),
            "not-a-uuid",
            "",
        ):
            with pytest.raises(ToolError, match="canonical UUID"):
                await unlocked_server.call_tool(
                    "qdrant_store",
                    {
                        "information": "x",
                        "collection_name": "pid_reject_test",
                        "point_id": bad,
                    },
                )


class TestBulkStoreWithPointIds:
    async def test_bulk_each_retrievable_by_its_id(self, unlocked_server):
        ids = [canonical(f"session:{i}") for i in range(3)]
        result = tool_json(
            await unlocked_server.call_tool(
                "qdrant_bulk_store",
                {
                    "documents": [f"doc {i}" for i in range(3)],
                    "collection_name": "pid_bulk_test",
                    "point_ids": ids,
                },
            )
        )
        assert result.get("stored_count") == 3, result
        # The failure-attribution keys are strictly conditional — an
        # all-success run must not carry them (not even as 0/[]).
        assert "failed_count" not in result, result
        assert "failed_point_ids" not in result, result
        for i, pid in enumerate(ids):
            payload = tool_json(
                await unlocked_server.call_tool(
                    "qdrant_get_point",
                    {"point_id": pid, "collection_name": "pid_bulk_test"},
                )
            )
            assert "error" not in payload, payload
            assert payload["id"] == pid  # bulk IDs round-trip byte-identically
            assert f"doc {i}" in str(payload.get("payload", {}))

    async def test_ids_align_across_batches(self, unlocked_server):
        # batch_size=2 over 5 docs → 3 batches; each ID must stay glued to
        # its document across the batch-slice math (i is the batch START
        # index — an off-by-one here silently mislabels documents).
        ids = [canonical(f"batchdoc:{i}") for i in range(5)]
        result = tool_json(
            await unlocked_server.call_tool(
                "qdrant_bulk_store",
                {
                    "documents": [f"batch doc {i}" for i in range(5)],
                    "collection_name": "pid_batch_test",
                    "point_ids": ids,
                    "batch_size": 2,
                },
            )
        )
        assert result.get("stored_count") == 5, result
        for i, pid in enumerate(ids):
            payload = tool_json(
                await unlocked_server.call_tool(
                    "qdrant_get_point",
                    {"point_id": pid, "collection_name": "pid_batch_test"},
                )
            )
            assert f"batch doc {i}" in str(payload.get("payload", {})), (i, payload)

    async def test_length_mismatch_rejected(self, unlocked_server):
        with pytest.raises(ToolError, match="length must match"):
            await unlocked_server.call_tool(
                "qdrant_bulk_store",
                {
                    "documents": ["a", "b"],
                    "collection_name": "pid_len_test",
                    "point_ids": [canonical("only-one")],
                },
            )

    async def test_non_canonical_element_rejected(self, unlocked_server):
        with pytest.raises(ToolError, match="canonical UUID"):
            await unlocked_server.call_tool(
                "qdrant_bulk_store",
                {
                    "documents": ["a", "b"],
                    "collection_name": "pid_badform_test",
                    "point_ids": [canonical("good"), uuid.uuid4().hex],
                },
            )

    async def test_duplicate_ids_rejected_including_cross_batch(self, unlocked_server):
        # The dup pair sits at indices 0 and 2 with batch_size=1 — different
        # upsert batches, so a per-batch check would MISS it and the second
        # write would silently collapse onto the first (last write wins).
        # The whole-list pre-batching check must reject it.
        dup = canonical("collide")
        with pytest.raises(ToolError, match="duplicate"):
            await unlocked_server.call_tool(
                "qdrant_bulk_store",
                {
                    "documents": ["a", "b", "c"],
                    "collection_name": "pid_dup_test",
                    "point_ids": [dup, canonical("fine"), dup],
                    "batch_size": 1,
                },
            )

    async def test_omitted_point_ids_behaves_as_today(self, unlocked_server):
        result = tool_json(
            await unlocked_server.call_tool(
                "qdrant_bulk_store",
                {
                    "documents": ["plain 1", "plain 2"],
                    "collection_name": "pid_plain_bulk_test",
                },
            )
        )
        assert result.get("stored_count") == 2, result
        # Identity-shaped, not just count-shaped: two DISTINCT points must
        # exist (a regression hoisting the uuid4 mint out of the per-point
        # loop would collapse all docs onto one ID yet keep stored_count).
        info = await unlocked_server.qdrant_connector.get_collection_info(
            "pid_plain_bulk_test"
        )
        assert info.get("points_count") == 2, info


class TestLockInteraction:
    async def test_prefix_rejection_wins_over_point_id_validation(self, locked_server):
        # Guard precedence: the namespace rejection must fire BEFORE the
        # point_id form check (the guard is contractually the first
        # statement; PR-B's validation sits below it).
        with pytest.raises(ToolError, match="not allowed on this endpoint"):
            await locked_server.call_tool(
                "qdrant_store",
                {
                    "information": "x",
                    "collection_name": "other_project",
                    "point_id": "not-even-a-uuid",
                },
            )

    async def test_point_id_works_on_admitted_collection(self, locked_server):
        pid = canonical("locked:doc")
        await locked_server.call_tool(
            "qdrant_store",
            {
                "information": "locked archive doc",
                "collection_name": "mld_pid_test",
                "point_id": pid,
            },
        )
        payload = tool_json(
            await locked_server.call_tool(
                "qdrant_get_point",
                {"point_id": pid, "collection_name": "mld_pid_test"},
            )
        )
        assert payload["id"] == pid


class TestBulkPartialFailure:
    async def test_failed_batch_ids_attributed_and_later_batches_stay_aligned(
        self, unlocked_server, monkeypatch
    ):
        # THE load-bearing slice invariant: batch_ids uses the ABSOLUTE batch
        # start index i, so a FAILED (skipped) earlier batch must not shift
        # later batches' IDs. In an all-success run an accumulator-indexing
        # regression (point_ids[total_stored:...]) is byte-identical, so only
        # a failing batch can falsify it. Fail exactly batch #2 (of 5
        # single-doc batches) via an upsert spy.
        ids = [canonical(f"failtest:{i}") for i in range(5)]
        connector = unlocked_server.qdrant_connector
        real_upsert = connector._client.upsert
        calls = {"n": 0}

        async def flaky_upsert(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("injected batch failure")
            return await real_upsert(*args, **kwargs)

        monkeypatch.setattr(connector._client, "upsert", flaky_upsert)
        result = tool_json(
            await unlocked_server.call_tool(
                "qdrant_bulk_store",
                {
                    "documents": [f"fail doc {i}" for i in range(5)],
                    "collection_name": "pid_fail_test",
                    "point_ids": ids,
                    "batch_size": 1,
                },
            )
        )
        assert result.get("stored_count") == 4, result
        # The DOCUMENTED contract: success stays True on partial failure
        # (consumers key off the attribution keys, not the flag).
        assert result.get("success") is True, result
        # Attribution: the failed batch's exact ID is named, nothing else.
        assert result.get("failed_count") == 1, result
        assert result.get("failed_point_ids") == [ids[1]], result
        # Alignment: every SURVIVING ID maps to its own document...
        for i in (0, 2, 3, 4):
            payload = tool_json(
                await unlocked_server.call_tool(
                    "qdrant_get_point",
                    {"point_id": ids[i], "collection_name": "pid_fail_test"},
                )
            )
            assert f"fail doc {i}" in str(payload.get("payload", {})), (i, payload)
        # ...and the failed ID is genuinely absent (not-found error payload).
        missing = tool_json(
            await unlocked_server.call_tool(
                "qdrant_get_point",
                {"point_id": ids[1], "collection_name": "pid_fail_test"},
            )
        )
        assert "error" in missing, missing

    async def test_failed_batch_without_point_ids_attributes_count_only(
        self, unlocked_server, monkeypatch
    ):
        # The DEFAULT (server-minted) path through the NEW failure handler —
        # what every pre-existing consumer hits on a batch failure: the call
        # must still continue-not-raise, carry failed_count, and NOT carry a
        # failed_point_ids key (absent, never an empty list).
        connector = unlocked_server.qdrant_connector
        real_upsert = connector._client.upsert
        calls = {"n": 0}

        async def flaky_upsert(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("injected batch failure")
            return await real_upsert(*args, **kwargs)

        monkeypatch.setattr(connector._client, "upsert", flaky_upsert)
        result = tool_json(
            await unlocked_server.call_tool(
                "qdrant_bulk_store",
                {
                    "documents": [f"noid doc {i}" for i in range(5)],
                    "collection_name": "pid_noid_fail_test",
                    "batch_size": 1,
                },
            )
        )
        assert result.get("success") is True, result
        assert result.get("stored_count") == 4, result
        assert result.get("failed_count") == 1, result
        assert "failed_point_ids" not in result, result

    async def test_multi_doc_failed_batch_attributes_whole_slice(
        self, unlocked_server, monkeypatch
    ):
        # batch_size=2 over 4 docs; fail batch #2 → BOTH of its IDs must be
        # attributed, order-preserving (whole-slice extend, not singleton).
        ids = [canonical(f"slice:{i}") for i in range(4)]
        connector = unlocked_server.qdrant_connector
        real_upsert = connector._client.upsert
        calls = {"n": 0}

        async def flaky_upsert(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("injected batch failure")
            return await real_upsert(*args, **kwargs)

        monkeypatch.setattr(connector._client, "upsert", flaky_upsert)
        result = tool_json(
            await unlocked_server.call_tool(
                "qdrant_bulk_store",
                {
                    "documents": [f"slice doc {i}" for i in range(4)],
                    "collection_name": "pid_slice_fail_test",
                    "point_ids": ids,
                    "batch_size": 2,
                },
            )
        )
        assert result.get("stored_count") == 2, result
        assert result.get("failed_count") == 2, result
        assert result.get("failed_point_ids") == [ids[2], ids[3]], result


class TestConnectorDirectGuards:
    """The connector's own fail-loud guards — a DIRECT caller bypasses the
    tool layer, so these cannot be proven through call_tool (the tool guard
    shadows them; deleting the connector guard would leave call_tool tests
    green while restoring the IndexError-swallowed-into-success no-op)."""

    async def test_bulk_length_mismatch_raises_at_connector(self, unlocked_server):
        with pytest.raises(ValueError, match="entries length"):
            await unlocked_server.qdrant_connector.bulk_store(
                entries=[Entry(content="a"), Entry(content="b")],
                collection_name="direct_guard_test",
                point_ids=[canonical("only-one")],
            )

    async def test_bulk_empty_entries_with_stray_ids_raises(self, unlocked_server):
        # The length check sits BEFORE the empty-entries early return — a
        # stray point_ids on an empty call is a caller bug that must fail
        # loud, not silently no-op with stored_count 0.
        with pytest.raises(ValueError, match="entries length"):
            await unlocked_server.qdrant_connector.bulk_store(
                entries=[],
                collection_name="direct_guard_test",
                point_ids=[canonical("stray")],
            )

    async def test_store_empty_string_never_silently_mints(self, unlocked_server):
        # store() deliberately uses `is not None` (not `or`): an empty-string
        # point_id must NEVER silently mint a server ID. On :memory: the ""
        # reaches PointStruct and fails UUID validation loudly — a regression
        # to `or` would mint and store successfully, turning this red.
        with pytest.raises(Exception):
            await unlocked_server.qdrant_connector.store(
                Entry(content="x"),
                collection_name="direct_guard_test",
                point_id="",
            )


class TestBulkEdgeBoundaries:
    async def test_empty_documents_with_empty_point_ids_is_clean_noop(
        self, unlocked_server
    ):
        result = tool_json(
            await unlocked_server.call_tool(
                "qdrant_bulk_store",
                {
                    "documents": [],
                    "collection_name": "pid_empty_test",
                    "point_ids": [],
                },
            )
        )
        assert result.get("stored_count") == 0, result

    async def test_empty_documents_with_stray_point_ids_rejected(self, unlocked_server):
        with pytest.raises(ToolError, match="length must match"):
            await unlocked_server.call_tool(
                "qdrant_bulk_store",
                {
                    "documents": [],
                    "collection_name": "pid_empty_test",
                    "point_ids": [canonical("stray")],
                },
            )

    async def test_metadata_list_and_point_ids_stay_aligned(self, unlocked_server):
        # The two independently-indexed parallel lists must stay glued to the
        # same document (the realistic eviction shape: derived ID + metadata).
        ids = [canonical(f"both:{i}") for i in range(3)]
        metas = [{"idx": i} for i in range(3)]
        await unlocked_server.call_tool(
            "qdrant_bulk_store",
            {
                "documents": [f"both doc {i}" for i in range(3)],
                "collection_name": "pid_both_test",
                "point_ids": ids,
                "metadata_list": metas,
            },
        )
        for i, pid in enumerate(ids):
            payload = tool_json(
                await unlocked_server.call_tool(
                    "qdrant_get_point",
                    {"point_id": pid, "collection_name": "pid_both_test"},
                )
            )
            body = str(payload.get("payload", {}))
            assert f"both doc {i}" in body, (i, payload)
            assert f"'idx': {i}" in body or f'"idx": {i}' in body, (i, payload)

    async def test_bulk_prefix_rejection_wins_over_point_id_validation(
        self, monkeypatch
    ):
        from tests.conftest import make_server

        server = make_server(monkeypatch, prefixes="mld_")
        with pytest.raises(ToolError, match="not allowed on this endpoint"):
            await server.call_tool(
                "qdrant_bulk_store",
                {
                    "documents": ["x"],
                    "collection_name": "other_project",
                    "point_ids": ["not-even-a-uuid"],
                },
            )


class TestConfirmContractWireShape:
    async def test_get_point_miss_is_iserror_false_with_error_key(
        self, unlocked_server
    ):
        # The documented read-back-confirm inversion hazard: a MISS comes back
        # as isError=FALSE carrying an error-key payload — a confirm gate that
        # keys on call success would read a miss as "confirmed". Pin the wire
        # shape at the protocol layer.
        import json as _json

        from mcp.shared.memory import (
            create_connected_server_and_client_session as connect,
        )

        async with connect(unlocked_server._mcp_server) as session:
            r = await session.call_tool(
                "qdrant_get_point",
                {
                    "point_id": canonical("never:stored"),
                    "collection_name": "pid_miss_test",
                },
            )
            assert r.isError is False
            payload = _json.loads(r.content[0].text)
            assert "error" in payload, payload


class TestUpdateByClientId:
    async def test_store_update_get_workflow(self, unlocked_server):
        # The natural deterministic-addressing companion: store(pid) →
        # update_payload([pid]) → get_point(pid) reflects the merge.
        pid = canonical("update:me")
        await unlocked_server.call_tool(
            "qdrant_store",
            {
                "information": "updatable doc",
                "collection_name": "pid_upd_test",
                "point_id": pid,
            },
        )
        result = tool_json(
            await unlocked_server.call_tool(
                "qdrant_update_payload",
                {
                    "point_ids": [pid],
                    "payload": {"status": "processed"},
                    "collection_name": "pid_upd_test",
                    "key": "metadata",
                },
            )
        )
        assert result.get("success") is True, result
        payload = tool_json(
            await unlocked_server.call_tool(
                "qdrant_get_point",
                {"point_id": pid, "collection_name": "pid_upd_test"},
            )
        )
        assert "processed" in str(payload.get("payload", {})), payload


class TestDeleteByClientId:
    async def test_store_delete_get_workflow(self, unlocked_server):
        # The docs advertise delete-by-client-ID without a search — pin the
        # whole workflow: store(pid) → delete_points([pid]) → get_point(pid)
        # reports not-found.
        pid = canonical("delete:me")
        await unlocked_server.call_tool(
            "qdrant_store",
            {
                "information": "doomed doc",
                "collection_name": "pid_del_test",
                "point_id": pid,
            },
        )
        result = tool_json(
            await unlocked_server.call_tool(
                "qdrant_delete_points",
                {"point_ids": [pid], "collection_name": "pid_del_test"},
            )
        )
        assert result.get("success") is True, result
        payload = tool_json(
            await unlocked_server.call_tool(
                "qdrant_get_point",
                {"point_id": pid, "collection_name": "pid_del_test"},
            )
        )
        assert "error" in payload, payload
