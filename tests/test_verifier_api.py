"""Regression tests for ``POST /interactions/verify``.

Covers: effective parameter merging (path-level vs operation-level
override by ``(in, name)``), required/type/enum/readOnly/writeOnly
checks on parameters and bodies, response selection (exact / ``nXX`` /
default), representation selection by request Content-Type and Accept,
direction-aware body schema validation, 400 envelope validation, 422
reference/media/response-selection failures with no partial result,
stable error ordering by source JSON Pointer and input non-mutation.
"""

from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def make_doc(
    *,
    openapi="3.0.3",
    parameters=None,
    op_parameters=None,
    request_body=None,
    responses=None,
    components=None,
    path="/pets",
    method="post",
    path_item_extra=None,
    operation_extra=None,
):
    operation: dict = {}
    if op_parameters is not None:
        operation["parameters"] = op_parameters
    if request_body is not None:
        operation["requestBody"] = request_body
    operation["responses"] = (
        responses if responses is not None else {"200": {"description": "ok"}}
    )
    if operation_extra:
        operation.update(operation_extra)
    item = {method: operation}
    if parameters is not None:
        item["parameters"] = parameters
    if path_item_extra:
        item.update(path_item_extra)
    doc = {
        "openapi": openapi,
        "info": {"title": "Pet", "version": "1.0"},
        "paths": {path: item},
    }
    if components is not None:
        doc["components"] = components
    return doc


def call(doc, *, selector="POST /pets", request=None, response=None, extra=None):
    payload = {"document": doc, "selector": selector}
    if request is not None:
        payload["request"] = request
    if response is not None:
        payload["response"] = response
    if extra:
        payload.update(extra)
    return client.post("/interactions/verify", json=payload)


def ok_response(**kwargs):
    record = {"status_code": 200}
    record.update(kwargs)
    return record


PET_SCHEMA = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "name": {"type": "string"},
        "tag": {"type": "string", "enum": ["cat", "dog"]},
        "id": {"type": "integer", "readOnly": True},
    },
}


def json_body(schema, required=True):
    return {"required": required, "content": {"application/json": {"schema": schema}}}


def json_response(schema, code="200"):
    return {code: {"description": "ok", "content": {"application/json": {"schema": schema}}}}


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_valid_minimal_exchange():
    doc = make_doc()
    r = call(doc, response=ok_response())
    assert r.status_code == 200
    assert r.json() == {"valid": True, "error_count": 0, "errors": []}


def test_valid_full_exchange():
    doc = make_doc(
        parameters=[
            {"name": "limit", "in": "query", "schema": {"type": "integer"}},
            {"name": "X-Tenant", "in": "header", "schema": {"type": "string"}},
        ],
        request_body=json_body(PET_SCHEMA),
        responses=json_response(
            {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "integer", "readOnly": True}},
            }
        ),
    )
    r = call(
        doc,
        request={
            "query": {"limit": 10},
            "header": {"x-tenant": "demo"},
            "content_type": "application/json",
            "accept": "application/json",
            "body": {"name": "rex", "tag": "dog"},
        },
        response=ok_response(
            content_type="application/json", body={"id": 7}
        ),
    )
    assert r.status_code == 200
    assert r.json()["valid"] is True
    assert r.json()["errors"] == []


def test_request_record_defaults_to_empty():
    doc = make_doc()
    r = call(doc, response=ok_response())
    assert r.status_code == 200
    assert r.json()["valid"] is True


# --------------------------------------------------------------------------- #
# Parameter checks: required / type / enum / override
# --------------------------------------------------------------------------- #


def test_missing_required_query_parameter():
    doc = make_doc(
        op_parameters=[
            {"name": "limit", "in": "query", "required": True,
             "schema": {"type": "integer"}}
        ]
    )
    r = call(doc, response=ok_response())
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert body["errors"] == [
        {
            "side": "request",
            "pointer": "/paths/~1pets/post/parameters/0",
            "instance_pointer": "/request/query",
            "keyword": "required",
            "location": "query",
            "name": "limit",
            "reason": "missing required query parameter 'limit'",
        }
    ]


def test_path_parameter_is_always_required():
    doc = make_doc(
        path="/pets/{petId}",
        op_parameters=[
            {"name": "petId", "in": "path", "schema": {"type": "integer"}}
        ],
    )
    r = call(doc, selector="POST /pets/{petId}", response=ok_response())
    assert r.status_code == 200
    errors = r.json()["errors"]
    assert len(errors) == 1
    assert errors[0]["keyword"] == "required"
    assert errors[0]["location"] == "path"
    assert errors[0]["name"] == "petId"


def test_null_value_counts_as_missing():
    doc = make_doc(
        op_parameters=[
            {"name": "limit", "in": "query", "required": True,
             "schema": {"type": "integer"}}
        ]
    )
    r = call(doc, request={"query": {"limit": None}}, response=ok_response())
    assert r.status_code == 200
    assert r.json()["errors"][0]["keyword"] == "required"


def test_parameter_type_and_enum_violations():
    doc = make_doc(
        op_parameters=[
            {"name": "limit", "in": "query", "schema": {"type": "integer"}},
            {"name": "sort", "in": "query",
             "schema": {"type": "string", "enum": ["asc", "desc"]}},
            {"name": "session", "in": "cookie", "schema": {"type": "string"}},
        ]
    )
    r = call(
        doc,
        request={
            "query": {"limit": "ten", "sort": "sideways"},
            "cookie": {"session": 42},
        },
        response=ok_response(),
    )
    assert r.status_code == 200
    errors = r.json()["errors"]
    assert r.json()["valid"] is False
    by_name = {e["name"]: e for e in errors}
    assert by_name["limit"]["keyword"] == "type"
    assert by_name["limit"]["instance_pointer"] == "/request/query/limit"
    assert by_name["sort"]["keyword"] == "enum"
    assert by_name["session"]["keyword"] == "type"
    assert by_name["session"]["location"] == "cookie"


def test_header_parameter_matches_case_insensitively():
    doc = make_doc(
        op_parameters=[
            {"name": "X-Token", "in": "header", "required": True,
             "schema": {"type": "string"}}
        ]
    )
    r = call(doc, request={"header": {"x-token": "abc"}}, response=ok_response())
    assert r.status_code == 200
    assert r.json()["valid"] is True


def test_operation_level_parameter_overrides_path_level():
    # Path-level: optional integer; operation-level override: required enum.
    doc = make_doc(
        parameters=[
            {"name": "mode", "in": "query",
             "schema": {"type": "integer"}}
        ],
        op_parameters=[
            {"name": "mode", "in": "query", "required": True,
             "schema": {"type": "string", "enum": ["a", "b"]}}
        ],
    )
    # Missing -> required by the operation-level override.
    r = call(doc, response=ok_response())
    assert r.json()["errors"][0]["keyword"] == "required"
    assert (
        r.json()["errors"][0]["pointer"] == "/paths/~1pets/post/parameters/0"
    )
    # The override schema (string enum), not the path-level integer, applies.
    r = call(doc, request={"query": {"mode": 3}}, response=ok_response())
    errors = r.json()["errors"]
    assert {e["keyword"] for e in errors} == {"type", "enum"}
    assert all(
        e["pointer"].startswith("/paths/~1pets/post/parameters/0")
        for e in errors
    )
    r = call(doc, request={"query": {"mode": "a"}}, response=ok_response())
    assert r.json()["valid"] is True


def test_path_level_parameter_applies_when_not_overridden():
    doc = make_doc(
        parameters=[
            {"name": "limit", "in": "query", "required": True,
             "schema": {"type": "integer"}}
        ],
    )
    r = call(doc, response=ok_response())
    assert r.json()["errors"][0]["keyword"] == "required"
    r = call(doc, request={"query": {"limit": 5}}, response=ok_response())
    assert r.json()["valid"] is True


def test_unknown_recorded_parameters_are_ignored():
    doc = make_doc()
    r = call(
        doc,
        request={"query": {"undeclared": "x"}, "header": {"X-Other": "y"}},
        response=ok_response(),
    )
    assert r.status_code == 200
    assert r.json()["valid"] is True


def test_parameter_ref_resolution_and_component_pointer():
    doc = make_doc(
        op_parameters=[{"$ref": "#/components/parameters/Limit"}],
        components={
            "parameters": {
                "Limit": {
                    "name": "limit",
                    "in": "query",
                    "required": True,
                    "schema": {"type": "integer"},
                }
            }
        },
    )
    r = call(doc, response=ok_response())
    errors = r.json()["errors"]
    assert errors[0]["pointer"] == "/components/parameters/Limit"
    r = call(doc, request={"query": {"limit": "x"}}, response=ok_response())
    errors = r.json()["errors"]
    assert errors[0]["keyword"] == "type"
    assert (
        errors[0]["pointer"] == "/components/parameters/Limit/schema"
    )


def test_parameter_object_value_readonly_flag():
    doc = make_doc(
        op_parameters=[
            {
                "name": "filter",
                "in": "query",
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "created": {"type": "string", "readOnly": True},
                    },
                },
            }
        ]
    )
    r = call(
        doc,
        request={"query": {"filter": {"status": "ok", "created": "today"}}},
        response=ok_response(),
    )
    errors = r.json()["errors"]
    assert len(errors) == 1
    assert errors[0]["keyword"] == "readOnly"
    assert errors[0]["instance_pointer"] == "/request/query/filter/created"


# --------------------------------------------------------------------------- #
# Body checks: direction-aware readOnly/writeOnly and required
# --------------------------------------------------------------------------- #


def test_request_body_schema_violations():
    doc = make_doc(request_body=json_body(PET_SCHEMA))
    r = call(
        doc,
        request={"content_type": "application/json",
                 "body": {"tag": "fish", "id": 3}},
        response=ok_response(),
    )
    errors = r.json()["errors"]
    keywords = {(e["keyword"], e["instance_pointer"]) for e in errors}
    assert ("required", "/request/body") in keywords
    assert ("enum", "/request/body/tag") in keywords
    assert ("readOnly", "/request/body/id") in keywords
    readonly = [e for e in errors if e["keyword"] == "readOnly"][0]
    assert "recorded request" in readonly["reason"]


def test_response_body_writeonly_and_type_violations():
    doc = make_doc(
        responses=json_response(
            {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "secret": {"type": "string", "writeOnly": True},
                },
            }
        )
    )
    r = call(
        doc,
        response=ok_response(body={"id": "abc", "secret": "s"}),
    )
    errors = r.json()["errors"]
    by_keyword = {e["keyword"]: e for e in errors}
    assert by_keyword["type"]["instance_pointer"] == "/response/body/id"
    assert by_keyword["writeOnly"]["instance_pointer"] == "/response/body/secret"
    assert "recorded response" in by_keyword["writeOnly"]["reason"]
    assert all(e["side"] == "response" for e in errors)


def test_direction_aware_required_exemptions():
    # readOnly member is required on responses only; writeOnly on requests.
    schema = {
        "type": "object",
        "required": ["id", "password"],
        "properties": {
            "id": {"type": "integer", "readOnly": True},
            "password": {"type": "string", "writeOnly": True},
        },
    }
    doc = make_doc(
        request_body=json_body(schema), responses=json_response(schema)
    )
    # Request may omit readOnly 'id'; response may omit writeOnly 'password'.
    r = call(
        doc,
        request={"content_type": "application/json",
                 "body": {"password": "x"}},
        response=ok_response(body={"id": 1}),
    )
    assert r.json()["valid"] is True
    # But the opposite omissions are violations.
    r = call(
        doc,
        request={"content_type": "application/json", "body": {"id": 1}},
        response=ok_response(body={"password": "x"}),
    )
    keywords = {(e["side"], e["keyword"]) for e in r.json()["errors"]}
    assert ("request", "readOnly") in keywords
    assert ("request", "required") in keywords  # password missing
    assert ("response", "writeOnly") in keywords
    assert ("response", "required") in keywords  # id missing


def test_required_request_body_missing():
    doc = make_doc(request_body=json_body(PET_SCHEMA, required=True))
    r = call(doc, response=ok_response())
    errors = r.json()["errors"]
    assert errors == [
        {
            "side": "request",
            "pointer": "/paths/~1pets/post/requestBody",
            "instance_pointer": "/request/body",
            "keyword": "required",
            "location": None,
            "name": None,
            "reason": (
                "the operation declares a required request body but the "
                "recorded request carries none"
            ),
        }
    ]


def test_optional_request_body_may_be_omitted():
    doc = make_doc(request_body=json_body(PET_SCHEMA, required=False))
    r = call(doc, response=ok_response())
    assert r.json()["valid"] is True


def test_response_body_absent_validated_as_null():
    doc = make_doc(responses=json_response({"type": "object"}))
    r = call(doc, response=ok_response())
    errors = r.json()["errors"]
    assert len(errors) == 1
    assert errors[0]["keyword"] == "type"
    assert errors[0]["instance_pointer"] == "/response/body"
    assert "got null" in errors[0]["reason"]


def test_nullable_difference_between_30_and_31():
    schema = {"type": "object", "nullable": True}
    doc30 = make_doc(responses=json_response(schema))
    r = call(doc30, response=ok_response())
    assert r.json()["valid"] is True  # 3.0 nullable admits null
    doc31 = make_doc(openapi="3.1.0", responses=json_response(schema))
    r = call(doc31, response=ok_response())
    assert r.json()["valid"] is False  # 3.1 ignores 'nullable'
    doc31u = make_doc(
        openapi="3.1.0",
        responses=json_response({"type": ["object", "null"]}),
    )
    r = call(doc31u, response=ok_response())
    assert r.json()["valid"] is True


# --------------------------------------------------------------------------- #
# Response selection: exact -> nXX -> default
# --------------------------------------------------------------------------- #


def _selection_doc():
    return make_doc(
        responses={
            "201": {"description": "created", "content": {"application/json": {
                "schema": {"type": "object", "required": ["exact"]}}}},
            "2XX": {"description": "class", "content": {"application/json": {
                "schema": {"type": "object", "required": ["wild"]}}}},
            "default": {"description": "other", "content": {"application/json": {
                "schema": {"type": "object", "required": ["fallback"]}}}},
        }
    )


def test_response_selected_by_exact_code():
    r = call(_selection_doc(), response={"status_code": 201, "body": {}})
    errors = r.json()["errors"]
    assert errors[0]["reason"] == "required property 'exact' is missing"
    assert "/responses/201/" in errors[0]["pointer"]


def test_response_selected_by_wildcard():
    r = call(_selection_doc(), response={"status_code": 202, "body": {}})
    errors = r.json()["errors"]
    assert errors[0]["reason"] == "required property 'wild' is missing"
    assert "/responses/2XX/" in errors[0]["pointer"]


def test_response_selected_by_default():
    r = call(_selection_doc(), response={"status_code": 500, "body": {}})
    errors = r.json()["errors"]
    assert errors[0]["reason"] == "required property 'fallback' is missing"
    assert "/responses/default/" in errors[0]["pointer"]


def test_response_status_code_as_string():
    r = call(_selection_doc(), response={"status_code": "201", "body": {"exact": 1}})
    assert r.json()["valid"] is True


def test_response_not_declared_is_422():
    doc = make_doc(responses={"200": {"description": "ok"}})
    r = call(doc, response={"status_code": 404})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "verification_failed"
    assert body["errors"][0]["code"] == "response_not_declared"
    assert body["errors"][0]["pointer"] == "/paths/~1pets/post/responses"
    assert "valid" not in body


# --------------------------------------------------------------------------- #
# Representation selection: request Content-Type / Accept
# --------------------------------------------------------------------------- #


def test_request_content_type_selects_representation():
    doc = make_doc(
        request_body={
            "required": True,
            "content": {
                "application/json": {"schema": {"type": "object"}},
                "text/plain": {"schema": {"type": "string"}},
            },
        }
    )
    r = call(
        doc,
        request={"content_type": "text/plain; charset=utf-8", "body": 42},
        response=ok_response(),
    )
    errors = r.json()["errors"]
    assert errors[0]["keyword"] == "type"
    assert errors[0]["reason"] == "expected type string, got integer"


def test_request_content_type_without_match_is_422():
    doc = make_doc(request_body=json_body(PET_SCHEMA))
    r = call(
        doc,
        request={"content_type": "text/csv", "body": {"name": "x"}},
        response=ok_response(),
    )
    assert r.status_code == 422
    errors = r.json()["errors"]
    assert errors[0]["code"] == "unsupported_media_type"
    assert errors[0]["supported"] == ["application/json"]


def test_body_without_content_type_is_422():
    doc = make_doc(request_body=json_body(PET_SCHEMA))
    r = call(doc, request={"body": {"name": "x"}}, response=ok_response())
    assert r.status_code == 422
    assert r.json()["errors"][0]["code"] == "unsupported_media_type"


def test_accept_selects_response_representation():
    doc = make_doc(
        responses={
            "200": {
                "description": "ok",
                "content": {
                    "application/json": {"schema": {"type": "object"}},
                    "text/plain": {"schema": {"type": "string"}},
                },
            }
        }
    )
    r = call(
        doc,
        request={"accept": "text/plain"},
        response=ok_response(body="hello"),
    )
    assert r.json()["valid"] is True
    r = call(
        doc,
        request={"accept": "text/plain"},
        response=ok_response(body={"a": 1}),
    )
    assert r.json()["errors"][0]["reason"] == "expected type string, got object"


def test_accept_without_match_is_422():
    doc = make_doc(responses=json_response({"type": "object"}))
    r = call(doc, request={"accept": "text/csv"}, response=ok_response(body={}))
    assert r.status_code == 422
    errors = r.json()["errors"]
    assert errors[0]["code"] == "not_acceptable"
    assert errors[0]["available"] == ["application/json"]


def test_empty_accept_rejects_declared_content():
    doc = make_doc(responses=json_response({"type": "object"}))
    r = call(doc, request={"accept": ""}, response=ok_response(body={}))
    assert r.status_code == 422
    assert r.json()["errors"][0]["code"] == "not_acceptable"


def test_missing_accept_means_any():
    doc = make_doc(responses=json_response({"type": "object"}))
    r = call(doc, response=ok_response(body={}))
    assert r.json()["valid"] is True


def test_accept_q_zero_is_not_selectable():
    doc = make_doc(responses=json_response({"type": "object"}))
    r = call(
        doc,
        request={"accept": "application/json;q=0"},
        response=ok_response(body={}),
    )
    assert r.status_code == 422
    assert r.json()["errors"][0]["code"] == "not_acceptable"


# --------------------------------------------------------------------------- #
# 422: references and contract structure (no partial results)
# --------------------------------------------------------------------------- #


def test_dangling_parameter_ref_is_422():
    doc = make_doc(
        op_parameters=[{"$ref": "#/components/parameters/Missing"}]
    )
    r = call(doc, response=ok_response())
    assert r.status_code == 422
    errors = r.json()["errors"]
    assert errors[0]["code"] == "unresolvable_reference"
    assert errors[0]["ref"] == "#/components/parameters/Missing"
    assert "valid" not in r.json()


def test_external_response_ref_is_422():
    doc = make_doc(
        responses={"200": {"$ref": "https://example.com/r.json#/R"}}
    )
    r = call(doc, response=ok_response())
    assert r.status_code == 422
    assert r.json()["errors"][0]["code"] == "unresolvable_reference"


def test_dangling_schema_ref_during_validation_is_422_without_partial():
    # The query parameter violation is a valid finding, but the dangling
    # schema reference in the response must turn the whole answer into a
    # 422 with no partial validation result.
    doc = make_doc(
        op_parameters=[
            {"name": "limit", "in": "query", "required": True,
             "schema": {"type": "integer"}}
        ],
        responses=json_response({"$ref": "#/components/schemas/Missing"}),
    )
    r = call(doc, response=ok_response(body={}))
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "verification_failed"
    assert body["errors"][0]["code"] == "unresolvable_reference"
    assert "valid" not in body


def test_component_schema_ref_validates_and_points_at_component():
    doc = make_doc(
        responses=json_response({"$ref": "#/components/schemas/Pet"}),
        components={
            "schemas": {
                "Pet": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            }
        },
    )
    r = call(doc, response=ok_response(body={"name": 5}))
    errors = r.json()["errors"]
    assert errors[0]["pointer"] == "/components/schemas/Pet/properties/name"


def test_circular_parameter_ref_is_422():
    doc = make_doc(
        op_parameters=[{"$ref": "#/components/parameters/A"}],
        components={
            "parameters": {
                "A": {"$ref": "#/components/parameters/B"},
                "B": {"$ref": "#/components/parameters/A"},
            }
        },
    )
    r = call(doc, response=ok_response())
    assert r.status_code == 422
    assert "circular" in r.json()["errors"][0]["error"]


def test_invalid_declared_media_type_is_422():
    doc = make_doc(
        responses={"200": {"description": "ok", "content": {"not a type": {}}}}
    )
    r = call(doc, response=ok_response())
    assert r.status_code == 422
    assert r.json()["errors"][0]["code"] == "invalid_media_type"


def test_missing_responses_object_is_422():
    doc = make_doc(operation_extra={"responses": {}})
    r = call(doc, response=ok_response())
    assert r.status_code == 422
    assert r.json()["errors"][0]["code"] == "invalid_operation"


def test_duplicate_parameter_definition_is_422():
    doc = make_doc(
        op_parameters=[
            {"name": "a", "in": "query", "schema": {"type": "string"}},
            {"name": "a", "in": "query", "schema": {"type": "string"}},
        ]
    )
    r = call(doc, response=ok_response())
    assert r.status_code == 422
    assert r.json()["errors"][0]["code"] == "duplicate_parameter"


def test_multiple_contract_problems_reported_together():
    doc = make_doc(
        op_parameters=[{"$ref": "#/components/parameters/Missing"}],
        responses={"200": {"description": "ok", "content": {"bad key": {}}}},
    )
    r = call(doc, response=ok_response())
    assert r.status_code == 422
    codes = {e["code"] for e in r.json()["errors"]}
    # The unparseable media type key also makes response representation
    # selection impossible, so every problem is reported at once.
    assert codes == {
        "unresolvable_reference",
        "invalid_media_type",
        "not_acceptable",
    }


# --------------------------------------------------------------------------- #
# 400: document, selector, envelope
# --------------------------------------------------------------------------- #


def test_invalid_document_is_400():
    r = call({"openapi": "2.0", "paths": {}}, response=ok_response())
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_openapi_document"


def test_malformed_selector_is_400():
    doc = make_doc()
    r = call(doc, selector="garbage", response=ok_response())
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_selector"


def test_unknown_path_or_method_is_400():
    doc = make_doc()
    r = call(doc, selector="POST /nope", response=ok_response())
    assert r.status_code == 400
    assert "path is not declared" in r.json()["messages"][0]
    r = call(doc, selector="GET /pets", response=ok_response())
    assert r.status_code == 400
    assert "operation is not declared" in r.json()["messages"][0]


def test_envelope_validation_is_400():
    doc = make_doc()
    # response missing entirely
    r = call(doc)
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_verify_request"
    # status_code missing / invalid
    r = call(doc, response={})
    assert r.status_code == 400
    assert "status_code" in r.json()["messages"][0]
    r = call(doc, response={"status_code": 99})
    assert r.status_code == 400
    r = call(doc, response={"status_code": "abc"})
    assert r.status_code == 400
    # unknown fields
    r = call(doc, request={"params": {}}, response=ok_response())
    assert r.status_code == 400
    assert "request.params" in r.json()["messages"][0]
    r = call(doc, response={"status_code": 200, "status": 200})
    assert r.status_code == 400
    assert "response.status" in r.json()["messages"][0]
    # group not an object
    r = call(doc, request={"query": "a=1"}, response=ok_response())
    assert r.status_code == 400
    # header name case collision
    r = call(doc, request={"header": {"X-A": "1", "x-a": "2"}},
             response=ok_response())
    assert r.status_code == 400
    assert "differ only in case" in r.json()["messages"][0]
    # malformed content types / accept
    r = call(doc, request={"content_type": "json"}, response=ok_response())
    assert r.status_code == 400
    r = call(doc, request={"accept": "text"}, response=ok_response())
    assert r.status_code == 400
    r = call(doc, response={"status_code": 200, "content_type": "*/*"})
    assert r.status_code == 400
    r = call(doc, response={"status_code": 200, "headers": ["a"]})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Ordering, stability, non-mutation
# --------------------------------------------------------------------------- #


def test_errors_sorted_by_source_pointer():
    doc = make_doc(
        op_parameters=[
            {"name": "b", "in": "query", "required": True,
             "schema": {"type": "string"}},
            {"name": "a", "in": "query", "required": True,
             "schema": {"type": "string"}},
        ],
        request_body=json_body(PET_SCHEMA),
        responses=json_response({"type": "object"}),
    )
    r = call(doc, response=ok_response())
    errors = r.json()["errors"]
    assert r.json()["valid"] is False
    pointers = [e["pointer"] for e in errors]
    assert pointers == sorted(pointers)
    # Both missing query parameters are reported, path-level order irrelevant.
    names = {e["name"] for e in errors if e["location"] == "query"}
    assert names == {"a", "b"}
    # Object key order inside the document does not change the outcome.
    doc2 = copy.deepcopy(doc)
    doc2["paths"]["/pets"]["post"]["parameters"] = [
        {"schema": p["schema"], "required": p["required"],
         "in": p["in"], "name": p["name"]}
        for p in doc2["paths"]["/pets"]["post"]["parameters"]
    ]
    r2 = call(doc2, response=ok_response())
    assert r2.json() == r.json()
    # Repeated calls with the same inputs are byte-identical.
    assert call(doc, response=ok_response()).json() == r.json()


def test_input_is_never_modified():
    doc = make_doc(
        op_parameters=[
            {"name": "limit", "in": "query", "required": True,
             "schema": {"type": "integer"}}
        ],
        request_body=json_body(PET_SCHEMA),
        responses=json_response({"type": "object"}),
    )
    request = {"query": {"limit": "x"}, "content_type": "application/json",
               "body": {"name": 1}}
    response = {"status_code": 200, "body": {"a": 1}}
    snapshot = copy.deepcopy(
        {"document": doc, "request": request, "response": response}
    )
    r = call(doc, request=request, response=response)
    assert r.status_code == 200
    assert {"document": doc, "request": request, "response": response} == snapshot


def test_openapi31_ref_siblings_merge_for_response():
    doc = make_doc(
        openapi="3.1.0",
        responses={
            "200": {
                "$ref": "#/components/responses/Base",
                "description": "override",
            }
        },
        components={
            "responses": {
                "Base": {
                    "description": "base",
                    "content": {"application/json": {
                        "schema": {"type": "object"}}},
                }
            }
        },
    )
    r = call(doc, response=ok_response(body="not-an-object"))
    errors = r.json()["errors"]
    assert errors[0]["keyword"] == "type"
    assert (
        errors[0]["pointer"]
        == "/components/responses/Base/content/application~1json/schema"
    )


def test_openapi30_ignores_ref_siblings():
    doc = make_doc(
        responses={
            "200": {
                "$ref": "#/components/responses/Base",
                "content": {"text/plain": {"schema": {"type": "string"}}},
            }
        },
        components={
            "responses": {
                "Base": {
                    "description": "base",
                    "content": {"application/json": {
                        "schema": {"type": "object"}}},
                }
            }
        },
    )
    # 3.0: sibling 'content' ignored -> application/json from the target.
    r = call(doc, response=ok_response(body={}))
    assert r.json()["valid"] is True
