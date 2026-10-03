"""Regression tests for ``POST /exchanges/verify``.

Covers: request parameter required/type/enum checks (path/query/header/
cookie), path-level vs operation-level parameter overrides, case-
insensitive header matching, direction markers on parameters
(readOnly request / writeOnly response), request body presence and
Content-Type selection (parameter containment, wildcard specificity,
charset), request/response JSON schema validation with direction-aware
required/readOnly/writeOnly, non-JSON representation skip, response key
selection (exact / nXX / default, int and string codes), response
headers (incl. ``$ref``), actual-response Content-Type matching and
Accept fallback/422, local ``$ref`` resolution of parameters,
requestBody, response and headers (multi-hop, dangling / external /
non-object), recursive schemas, 400 envelope validation, 422 contract
failures with no partial verdict, stable source-pointer ordering and
input non-mutation.
"""

from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


PET_SCHEMA = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "id": {"type": "integer", "readOnly": True},
        "password": {"type": "string", "writeOnly": True},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}


def make_doc(*, openapi="3.0.3", components=None, operation=None, path="/pets"):
    operation = operation or {
        "requestBody": {"required": True, "content": {
            "application/json": {"schema": {"$ref": "#/components/schemas/Pet"}},
        }},
        "responses": {
            "201": {"description": "created", "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/Pet"}},
            }},
            "4XX": {"description": "problem", "content": {
                "application/problem+json": {"schema": {
                    "type": "object",
                    "required": ["code"],
                    "properties": {"code": {"type": "string"}},
                }},
            }},
            "default": {"description": "unexpected"},
        },
    }
    doc = {
        "openapi": openapi,
        "info": {"title": "Pet", "version": "1.0"},
        "paths": {path: {"post": operation}},
    }
    doc["components"] = components or {"schemas": {"Pet": PET_SCHEMA}}
    return doc


def call(doc, request=None, response=None, *, selector="POST /pets"):
    payload = {"document": doc, "selector": selector,
               "response": response or {"status_code": 201}}
    if request is not None:
        payload["request"] = request
    return client.post("/exchanges/verify", json=payload)


def valid_request_body(**overrides):
    body = {"name": "Rex", "password": "secret"}
    body.update(overrides)
    return {"content_type": "application/json", "body": body}


# --------------------------------------------------------------------------- #
# Happy path / selection metadata
# --------------------------------------------------------------------------- #


def test_valid_exchange_reports_true_and_selection_metadata():
    r = call(
        make_doc(),
        valid_request_body(),
        {"status_code": "201", "content_type": "application/json; charset=utf-8",
         "body": {"id": 1, "name": "Rex", "tags": ["a"]}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["valid"] is True
    assert body["error_count"] == 0
    assert body["errors"] == []
    assert body["operation_pointer"] == "/paths/~1pets/post"
    assert body["request"] == {
        "present": True,
        "declared_media_type": "application/json",
        "media_type_pointer": (
            "/paths/~1pets/post/requestBody/content/application~1json"
        ),
        "json": True,
    }
    resp = body["response"]
    assert resp["status_code"] == 201
    assert resp["key"] == "201"
    assert resp["match"] == "exact"
    assert resp["present"] is True
    assert resp["declared_media_type"] == "application/json"
    assert resp["json"] is True
    assert resp["response_pointer"] == "/paths/~1pets/post/responses/201"


def test_wildcard_and_default_response_selection():
    r = call(
        make_doc(),
        valid_request_body(),
        {"status_code": 404, "content_type": "application/problem+json",
         "accept": "application/problem+json", "body": {"code": "nf"}},
    )
    assert r.status_code == 200, r.text
    resp = r.json()["response"]
    assert resp["key"] == "4XX"
    assert resp["match"] == "wildcard"
    assert resp["declared_media_type"] == "application/problem+json"

    r = call(make_doc(), valid_request_body(), {"status_code": 500})
    assert r.status_code == 200, r.text
    resp = r.json()["response"]
    assert resp["key"] == "default"
    assert resp["match"] == "default"
    assert resp["present"] is False
    assert resp["declared_media_type"] is None
    assert resp["json"] is False


def test_optional_request_body_omitted_is_valid():
    doc = make_doc(operation={
        "requestBody": {"required": False, "content": {
            "application/json": {"schema": {"type": "object"}}}},
        "responses": {"200": {"description": "ok"}},
    })
    r = call(doc, {}, {"status_code": 200})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["valid"] is True
    assert body["request"]["present"] is False
    assert body["request"]["declared_media_type"] is None


def test_non_json_request_representation_skips_body_schema():
    doc = make_doc(operation={
        "requestBody": {"required": True, "content": {
            "text/plain": {"schema": {"type": "string"}}}},
        "responses": {"204": {"description": "empty"}},
    })
    r = call(
        doc,
        {"content_type": "text/plain", "body": {"not": "a string"}},
        {"status_code": 204},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["valid"] is True
    assert body["request"]["json"] is False
    assert body["response"]["present"] is False


# --------------------------------------------------------------------------- #
# Request parameters
# --------------------------------------------------------------------------- #


def test_parameter_required_type_enum_checks():
    doc = make_doc(operation={
        "parameters": [
            {"name": "id", "in": "path", "schema": {"type": "integer"}},
            {"name": "limit", "in": "query", "required": True,
             "schema": {"type": "integer", "enum": [10, 20]}},
            {"name": "X-Trace", "in": "header",
             "schema": {"type": "string", "enum": ["a", "b"]}},
            {"name": "session", "in": "cookie",
             "schema": {"type": "string"}},
        ],
        "responses": {"200": {"description": "ok"}},
    })
    r = call(
        doc,
        {"path": {"id": "x"}, "query": {"limit": 30},
         "header": {"x-trace": "z"}, "cookie": {"session": 7}},
        {"status_code": 200},
        selector="POST /pets",
    )
    assert r.status_code == 200, r.text
    errors = r.json()["errors"]
    assert [e["keyword"] for e in errors] == ["type", "enum", "enum", "type"]
    by_name = {e["name"]: e for e in errors}
    assert by_name["id"]["location"] == "path"
    assert by_name["id"]["source_pointer"].endswith("/parameters/0/schema")
    assert by_name["limit"]["location"] == "query"
    assert by_name["X-Trace"]["location"] == "header"
    assert by_name["session"]["location"] == "cookie"
    for error in errors:
        assert error["side"] == "request"
        assert error["instance_pointer"] == ""


def test_missing_required_parameter_is_violation_optional_null_skipped():
    doc = make_doc(operation={
        "parameters": [
            {"name": "q", "in": "query", "required": True,
             "schema": {"type": "string"}},
            {"name": "o", "in": "query", "schema": {"type": "string"}},
        ],
        "responses": {"200": {"description": "ok"}},
    })
    r = call(doc, {"query": {"o": None}}, {"status_code": 200})
    assert r.status_code == 200
    errors = r.json()["errors"]
    assert len(errors) == 1
    assert errors[0]["keyword"] == "required"
    assert errors[0]["name"] == "q"
    assert errors[0]["instance_pointer"] == ""


def test_path_level_parameters_overridden_by_operation_level():
    doc = make_doc(operation={
        "parameters": [
            {"name": "p", "in": "query", "required": False,
             "schema": {"type": "string"}},
        ],
        "responses": {"200": {"description": "ok"}},
    })
    doc["paths"]["/pets"]["parameters"] = [
        {"name": "p", "in": "query", "required": True,
         "schema": {"type": "integer"}},
    ]
    # Operation-level override makes it an optional string: omission OK.
    r = call(doc, {"query": {}}, {"status_code": 200})
    assert r.status_code == 200
    assert r.json()["valid"] is True
    r = call(doc, {"query": {"p": "hello"}}, {"status_code": 200})
    assert r.status_code == 200
    assert r.json()["valid"] is True


def test_read_only_parameter_forbidden_in_request():
    doc = make_doc(operation={
        "parameters": [{"name": "v", "in": "query",
                        "schema": {"type": "string", "readOnly": True}}],
        "responses": {"200": {"description": "ok"}},
    })
    r = call(doc, {"query": {"v": "a"}}, {"status_code": 200})
    assert r.status_code == 200
    errors = r.json()["errors"]
    assert len(errors) == 1
    assert errors[0]["keyword"] == "readOnly"
    assert errors[0]["side"] == "request"


def test_parameter_ref_resolved_and_invalid_values_checked():
    doc = make_doc(
        components={
            "schemas": {"Pet": PET_SCHEMA},
            "parameters": {"Limit": {
                "name": "limit", "in": "query", "required": True,
                "schema": {"type": "integer"},
            }},
        },
        operation={
            "parameters": [{"$ref": "#/components/parameters/Limit"}],
            "responses": {"200": {"description": "ok"}},
        },
    )
    r = call(doc, {"query": {"limit": "no"}}, {"status_code": 200})
    assert r.status_code == 200
    errors = r.json()["errors"]
    assert [e["keyword"] for e in errors] == ["type"]
    assert errors[0]["source_pointer"] == (
        "/components/parameters/Limit/schema"
    )


# --------------------------------------------------------------------------- #
# Body schema validation
# --------------------------------------------------------------------------- #


def test_request_body_readonly_and_required_enforced_request_direction():
    r = call(
        make_doc(),
        {"content_type": "application/json",
         "body": {"id": 9, "password": "x"}},  # missing name, readOnly id
        {"status_code": "201"},
    )
    body = r.json()
    assert body["valid"] is False
    errors = body["errors"]
    keywords = {(e["keyword"], e["instance_pointer"]) for e in errors}
    assert ("required", "") in keywords
    assert ("readOnly", "/id") in keywords
    # writeOnly 'password' is allowed in a request.
    assert not any(e["keyword"] == "writeOnly" for e in errors)


def test_response_body_write_only_forbidden_and_readonly_allowed():
    r = call(
        make_doc(),
        valid_request_body(),
        {"status_code": 201, "content_type": "application/json",
         "body": {"id": 1, "name": "Rex", "password": "p"}},
    )
    errors = r.json()["errors"]
    assert len(errors) == 1
    assert errors[0]["side"] == "response"
    assert errors[0]["keyword"] == "writeOnly"
    assert errors[0]["instance_pointer"] == "/password"


def test_body_nested_type_and_item_violations():
    r = call(
        make_doc(),
        {"content_type": "application/json", "body": {"name": "Rex", "tags": ["ok", 3]}},
        {"status_code": "201"},
    )
    errors = r.json()["errors"]
    assert len(errors) == 1
    assert errors[0]["keyword"] == "type"
    assert errors[0]["side"] == "request"
    assert errors[0]["instance_pointer"] == "/tags/1"


# --------------------------------------------------------------------------- #
# Response headers
# --------------------------------------------------------------------------- #


def test_response_headers_required_type_enum():
    doc = make_doc(operation={
        "responses": {"200": {
            "description": "ok",
            "headers": {
                "X-Count": {"required": True, "schema": {"type": "integer"}},
                "X-Kind": {"schema": {"type": "string", "enum": ["a", "b"]}},
            },
        }},
    })
    r = call(doc, None, {"status_code": 200, "headers": {"x-kind": "z"}})
    errors = r.json()["errors"]
    assert {(e["keyword"], e["name"]) for e in errors} == {
        ("required", "X-Count"),
        ("enum", "X-Kind"),
    }
    for error in errors:
        assert error["side"] == "response"
        assert error["location"] == "header"


def test_response_header_ref_and_write_only_marker():
    doc = make_doc(
        components={
            "schemas": {"Pet": PET_SCHEMA},
            "headers": {
                "Count": {"required": True, "schema": {"type": "integer"}},
                "Secret": {"schema": {"type": "string", "writeOnly": True}},
            },
        },
        operation={"responses": {"200": {
            "description": "ok",
            "headers": {
                "X-Count": {"$ref": "#/components/headers/Count"},
                "X-Secret": {"$ref": "#/components/headers/Secret"},
            },
        }}},
    )
    r = call(
        doc, None,
        {"status_code": 200,
         "headers": {"x-count": "two", "x-secret": "leaked"}},
    )
    errors = r.json()["errors"]
    assert {(e["keyword"], e["source_pointer"]) for e in errors} == {
        ("type", "/components/headers/Count/schema"),
        ("writeOnly", "/components/headers/Secret/schema"),
    }


# --------------------------------------------------------------------------- #
# Media type selection
# --------------------------------------------------------------------------- #


def test_request_content_type_parameter_containment_and_wildcards():
    doc = make_doc(operation={
        "requestBody": {"required": True, "content": {
            "application/vnd.example+json; version=1": {
                "schema": {"type": "object"}},
            "application/*": {"schema": {"type": "object", "x": "wild"}},
        }},
        "responses": {"200": {"description": "ok"}},
    })
    r = call(
        doc,
        {"content_type": "application/vnd.example+json; version=1; charset=utf-8",
         "body": {}},
        {"status_code": 200},
    )
    assert r.status_code == 200, r.text
    assert r.json()["request"]["declared_media_type"] == (
        "application/vnd.example+json; version=1"
    )

    r = call(doc, {"content_type": "application/xml", "body": {}},
             {"status_code": 200})
    assert r.status_code == 200
    assert r.json()["request"]["declared_media_type"] == "application/*"


def test_unsupported_request_content_type_is_422():
    r = call(
        make_doc(),
        {"content_type": "text/csv", "body": {}},
        {"status_code": "201"},
    )
    assert r.status_code == 422
    payload = r.json()
    assert payload["error"] == "exchange_verification_failed"
    codes = [e["code"] for e in payload["errors"]]
    assert codes == ["unsupported_media_type"]
    assert "request" not in payload and "valid" not in payload


def test_required_body_missing_or_without_content_type_is_422():
    r = call(make_doc(), {}, {"status_code": "201"})
    assert r.status_code == 422
    assert [e["code"] for e in r.json()["errors"]] == ["request_body_missing"]

    r = call(make_doc(), {"body": {"name": "Rex"}}, {"status_code": "201"})
    assert r.status_code == 422
    assert [e["code"] for e in r.json()["errors"]] == [
        "request_body_content_type_missing"
    ]


def test_response_content_type_must_match_declared_and_accept():
    r = call(
        make_doc(), valid_request_body(),
        {"status_code": 201, "content_type": "text/html", "body": "<p>"},
    )
    assert r.status_code == 422
    assert [e["code"] for e in r.json()["errors"]] == [
        "response_media_type_not_declared"
    ]

    r = call(
        make_doc(), {**valid_request_body(), "accept": "text/html"},
        {"status_code": 201, "content_type": "application/json", "body": {}},
    )
    assert r.status_code == 422
    assert [e["code"] for e in r.json()["errors"]] == ["not_acceptable"]


def test_response_representation_falls_back_to_accept_without_content_type():
    # Body without a response Content-Type is selected through Accept.
    r = call(
        make_doc(),
        {**valid_request_body(), "accept": "application/json"},
        {"status_code": 201, "body": {"id": 1, "name": "Rex"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["response"]["declared_media_type"] == "application/json"

    # Empty Accept header rejects every representation -> 422.
    r = call(
        make_doc(),
        {**valid_request_body(), "accept": ""},
        {"status_code": 201, "body": {}},
    )
    assert r.status_code == 422
    assert [e["code"] for e in r.json()["errors"]] == [
        "response_media_type_missing"
    ]


# --------------------------------------------------------------------------- #
# Reference / structural failures (422, no partial verdict)
# --------------------------------------------------------------------------- #


def test_response_not_declared_is_422():
    doc = make_doc()
    doc["paths"]["/pets"]["post"]["responses"].pop("default")
    r = call(doc, valid_request_body(), {"status_code": 500})
    assert r.status_code == 422
    codes = [e["code"] for e in r.json()["errors"]]
    assert codes == ["response_not_declared"]
    assert "valid" not in r.json()


def test_external_reference_is_422():
    doc = make_doc(operation={
        "requestBody": {"$ref": "https://example.test/body"},
        "responses": {"200": {"description": "ok"}},
    })
    r = call(doc, {"content_type": "application/json", "body": {}},
             {"status_code": 200})
    assert r.status_code == 422
    assert [e["code"] for e in r.json()["errors"]] == [
        "unresolvable_reference"
    ]


def test_dangling_schema_reference_is_422_not_violation():
    doc = make_doc()
    doc["components"]["schemas"]["Pet"]["properties"]["name"] = {
        "$ref": "#/components/schemas/Missing"
    }
    r = call(
        doc, valid_request_body(),
        {"status_code": 201, "content_type": "application/json",
         "body": {"name": "Rex"}},
    )
    assert r.status_code == 422
    assert [e["code"] for e in r.json()["errors"]] == [
        "unresolvable_reference"
    ]


def test_dangling_response_header_reference_is_422():
    doc = make_doc(operation={"responses": {"200": {
        "description": "ok",
        "headers": {"H": {"$ref": "#/components/headers/Missing"}},
    }}})
    r = call(doc, None, {"status_code": 200, "headers": {"H": "x"}})
    assert r.status_code == 422
    assert [e["code"] for e in r.json()["errors"]] == [
        "unresolvable_reference"
    ]


def test_recursive_schema_validates_and_cycles_do_not_explode():
    node = {
        "type": "object",
        "properties": {"child": {"$ref": "#/components/schemas/Node"},
                       "name": {"type": "string"}},
    }
    doc = make_doc(
        components={"schemas": {"Node": node}},
        operation={
            "requestBody": {"required": True, "content": {
                "application/json": {"schema": {
                    "$ref": "#/components/schemas/Node"}}}},
            "responses": {"200": {"description": "ok"}},
        },
    )
    nested = {"name": "a", "child": {"name": "b", "child": {"name": 7}}}
    r = call(doc, {"content_type": "application/json", "body": nested},
             {"status_code": 200})
    assert r.status_code == 200
    errors = r.json()["errors"]
    assert len(errors) == 1
    assert errors[0]["instance_pointer"] == "/child/child/name"
    assert errors[0]["keyword"] == "type"


# --------------------------------------------------------------------------- #
# Envelope validation (400)
# --------------------------------------------------------------------------- #


def test_bad_document_selector_and_envelope_return_400():
    r = client.post("/exchanges/verify", json={
        "document": {"openapi": "2.0", "paths": {}},
        "selector": "POST /pets", "response": {"status_code": 200},
    })
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_openapi_document"

    r = call(make_doc(), None, {"status_code": 200}, selector="DELETE /missing")
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_selector"

    r = client.post("/exchanges/verify", json={
        "document": make_doc(), "selector": "POST /pets",
        "response": {"status_code": "nope"},
    })
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_exchange_request"

    r = client.post("/exchanges/verify", json={
        "document": make_doc(), "selector": "POST /pets",
        "request": {"content_type": "not a media type"},
        "response": {"status_code": 200},
    })
    assert r.status_code == 400

    r = client.post("/exchanges/verify", json={
        "document": make_doc(), "selector": "POST /pets",
        "request": {"query": "not-an-object"},
        "response": {"status_code": 200},
    })
    assert r.status_code == 400


def test_ambiguous_header_keys_return_400():
    r = call(
        make_doc(),
        {"header": {"X-Trace": "a", "x-trace": "b"}},
        {"status_code": "201"},
    )
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Ordering and non-mutation
# --------------------------------------------------------------------------- #


def test_errors_sorted_stably_by_source_pointer():
    doc = make_doc(operation={
        "parameters": [
            {"name": "z", "in": "query", "required": True,
             "schema": {"type": "integer"}},
            {"name": "a", "in": "query", "required": True,
             "schema": {"type": "integer"}},
        ],
        "responses": {"200": {"description": "ok"}},
    })
    r = call(doc, {"query": {"z": "z", "a": "a"}}, {"status_code": 200})
    errors = r.json()["errors"]
    # request violations precede response violations; within a side they
    # are ordered by source pointer (/parameters/0 < /parameters/1).
    assert [e["name"] for e in errors] == ["z", "a"]


def test_input_document_and_exchange_are_not_modified():
    doc = make_doc()
    snapshot = copy.deepcopy(doc)
    exchange_request = valid_request_body()
    exchange_response = {"status_code": 201, "content_type": "application/json",
                         "body": {"id": 1, "name": "Rex"}}
    req_snapshot = copy.deepcopy(exchange_request)
    call(doc, exchange_request, exchange_response)
    assert doc == snapshot
    assert exchange_request == req_snapshot
