"""Contract verification of one recorded joint-debugging exchange.

Implements ``POST /interactions/verify``: given an OpenAPI 3.x document,
a ``METHOD /path`` operation selector and one recorded request/response
pair (a joint-debugging record), check the record against the contract:

* the effective parameters are computed by merging path-level and
  operation-level ``parameters`` (an operation-level parameter overrides
  the path-level one sharing ``(in, name)``, duplicates within one level
  are rejected); every declared parameter is checked for presence when
  required (path parameters always are), and every supplied value is
  validated against the parameter's ``schema`` -- type, enum and the
  direction-aware ``readOnly`` / ``writeOnly`` rules included;
* the response is selected from the operation's ``responses`` by exact
  status code, then an ``nXX`` class wildcard (case-insensitive), then
  ``default``;
* the request representation is selected for the recorded request
  ``Content-Type`` and the response representation for the recorded
  ``Accept`` header (the same RFC 9110 negotiation as
  ``POST /representations/select``), then both recorded bodies are
  validated against the selected media type schemas -- the request body
  in request direction (``readOnly`` properties forbidden, ``writeOnly``
  members required) and the response body in response direction
  (``writeOnly`` forbidden, ``readOnly`` members required).

The endpoint answers ``200`` with ``valid`` and the request/response
errors sorted stably by their source JSON Pointer. Contract-level
problems -- unresolvable references, malformed parameter or media-type
declarations, a request ``Content-Type`` or ``Accept`` header that
selects no representation, or a status code no declared response covers
-- return ``422`` with every problem listed and no partial result. The
input document and records are never modified.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .callbacks import pointer_join
from .examples import (
    ExampleReferenceError,
    SchemaValidator,
    _RefIssues,
)
from .materializer import GROUPS, _build_parameters
from .negotiation import (
    MediaTypeParseError,
    _contract_error,
    _member_pointer,
    _parse_declared_media_type,
    _resolve_path_item,
    _resolve_referenced_object,
    normalize_status_code,
    parse_accept,
    parse_media_type,
    select_request_media_type,
    select_response_key,
    select_response_media_type,
)
from .refresolver import RefResolver
from .slicer import HTTP_METHODS, parse_selector, validate_openapi_shape


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class VerifyError(Exception):
    """Base class for verification request failures."""


class InvalidDocumentError(VerifyError):
    """The submitted document is not a recognizable OpenAPI 3.x doc (400)."""

    def __init__(self, messages: List[str]) -> None:
        self.messages = messages
        super().__init__("; ".join(messages))


class InvalidSelectorError(VerifyError):
    """The selector is malformed or unknown (400)."""

    def __init__(self, messages: List[str]) -> None:
        self.messages = messages
        super().__init__("; ".join(messages))


class InvalidVerifyRequestError(VerifyError):
    """The request/response record envelope is malformed (400)."""

    def __init__(self, messages: List[str]) -> None:
        self.messages = messages
        super().__init__("; ".join(messages))


class VerificationFailedError(VerifyError):
    """Contract-level problems blocking the verification (HTTP 422)."""

    def __init__(self, errors: List[Dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__(
            "; ".join(f"{e['code']} @ {e.get('pointer', '')}" for e in errors)
        )


# --------------------------------------------------------------------------- #
# Record envelope
# --------------------------------------------------------------------------- #

_REQUEST_FIELDS = (
    "path",
    "query",
    "header",
    "cookie",
    "content_type",
    "accept",
    "body",
)
_RESPONSE_FIELDS = ("status_code", "headers", "content_type", "body")


def _normalize_request_record(
    raw: Any,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Validate and normalize the recorded request envelope.

    A missing record means an empty request (no parameters, no headers,
    no body). Explicit JSON ``null`` values in parameter groups or the
    body are treated as "absent", mirroring the ``values`` convention of
    ``POST /requests/materialize``.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        return None, ["request must be a JSON object"]

    messages: List[str] = []
    for key in sorted(raw, key=str):
        if key not in _REQUEST_FIELDS:
            messages.append(
                f"request.{key}: unknown request record field; expected one "
                "of " + "/".join(_REQUEST_FIELDS)
            )

    record: Dict[str, Any] = {}
    for group in GROUPS:
        value = raw.get(group)
        if value is None:
            record[group] = {}
        elif not isinstance(value, dict):
            messages.append(
                f"request.{group} must be an object mapping parameter names "
                "to values"
            )
            record[group] = {}
        else:
            record[group] = value

    # Header names match case-insensitively, so two recorded keys that
    # differ only in case make the record ambiguous.
    seen_lower: Dict[str, str] = {}
    for key in record["header"]:
        if not isinstance(key, str):
            continue
        lowered = key.lower()
        if lowered in seen_lower and seen_lower[lowered] != key:
            messages.append(
                f"request.header: keys '{seen_lower[lowered]}' and '{key}' "
                "differ only in case; a recorded header name must be "
                "unambiguous"
            )
        else:
            seen_lower[lowered] = key

    content_type = raw.get("content_type")
    requested_media: Optional[Dict[str, Any]] = None
    if content_type is not None:
        if not isinstance(content_type, str):
            messages.append(
                "request.content_type must be a string when provided"
            )
        else:
            try:
                requested_media = parse_media_type(content_type)
            except MediaTypeParseError as exc:
                messages.append(str(exc))

    accept = raw.get("accept")
    # A missing Accept header means the client accepts everything.
    accept_ranges: List[Dict[str, Any]] = [
        {"type": "*", "subtype": "*", "parameters": {}, "q": 1.0}
    ]
    if accept is not None:
        if not isinstance(accept, str):
            messages.append("request.accept must be a string when provided")
        else:
            accept_ranges, accept_errors = parse_accept(accept)
            messages.extend(accept_errors)

    record["content_type"] = content_type if isinstance(content_type, str) else None
    record["requested_media"] = requested_media
    record["accept"] = accept if isinstance(accept, str) else None
    record["accept_ranges"] = accept_ranges
    record["body_present"] = raw.get("body") is not None
    record["body"] = raw.get("body")
    return record, messages


def _normalize_response_record(
    raw: Any,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Validate and normalize the recorded response envelope."""
    if raw is None:
        return None, [
            "response is required: supply the recorded response with its "
            "status_code"
        ]
    if not isinstance(raw, dict):
        return None, ["response must be a JSON object"]

    messages: List[str] = []
    for key in sorted(raw, key=str):
        if key not in _RESPONSE_FIELDS:
            messages.append(
                f"response.{key}: unknown response record field; expected "
                "one of " + "/".join(_RESPONSE_FIELDS)
            )

    code: Optional[int] = None
    if "status_code" not in raw:
        messages.append(
            "response.status_code is required (an integer 100..599 or a "
            "digit string)"
        )
    else:
        code, code_error = normalize_status_code(raw["status_code"])
        if code_error is not None:
            messages.append(code_error)

    headers = raw.get("headers")
    if headers is None:
        headers = {}
    elif not isinstance(headers, dict):
        messages.append(
            "response.headers must be an object mapping header names to values"
        )
        headers = {}

    content_type = raw.get("content_type")
    if content_type is not None:
        if not isinstance(content_type, str):
            messages.append(
                "response.content_type must be a string when provided"
            )
        else:
            try:
                parse_media_type(content_type)
            except MediaTypeParseError as exc:
                messages.append(str(exc))

    record = {
        "status_code": code,
        "headers": headers,
        "content_type": content_type if isinstance(content_type, str) else None,
        "body_present": raw.get("body") is not None,
        "body": raw.get("body"),
    }
    return record, messages


# --------------------------------------------------------------------------- #
# Contract-map checks and error helpers
# --------------------------------------------------------------------------- #


def _check_content_map(
    content: Dict[str, Any],
    content_pointer: str,
    errors: List[Dict[str, Any]],
) -> None:
    """Validate the keys and values of a requestBody/response content map."""
    for key, node in content.items():
        if not isinstance(key, str):
            errors.append(
                _contract_error(
                    "invalid_media_type",
                    content_pointer,
                    "content keys must be media type strings",
                )
            )
        elif not isinstance(node, dict):
            errors.append(
                _contract_error(
                    "invalid_operation",
                    pointer_join(content_pointer, key),
                    f"media type object for {key!r} must be an object",
                )
            )
        else:
            try:
                _parse_declared_media_type(key)
            except MediaTypeParseError as exc:
                errors.append(
                    _contract_error(
                        "invalid_media_type",
                        pointer_join(content_pointer, key),
                        str(exc),
                    )
                )


def _sorted_contract_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable, order-independent ordering of collected contract errors."""
    return sorted(
        errors,
        key=lambda error: (
            str(error.get("code", "")),
            str(error.get("pointer", "")),
            str(error.get("ref", "")),
            str(error.get("error", "")),
        ),
    )


def _lookup_value(
    group_values: Dict[str, Any], location: str, name: str
) -> Tuple[bool, Any]:
    """Find a declared parameter's recorded value (headers case-insensitive).

    An explicit JSON ``null`` counts as "no value", so a required
    parameter then misses its mandatory value and an optional one is
    skipped.
    """
    if location == "header":
        lowered = name.lower()
        for key, value in group_values.items():
            if isinstance(key, str) and key.lower() == lowered:
                return value is not None, value
        return False, None
    if name in group_values:
        value = group_values[name]
        return value is not None, value
    return False, None


def _run_schema_validation(
    document: Dict[str, Any],
    resolver: RefResolver,
    issues: _RefIssues,
    side: str,
    schema: Any,
    schema_pointer: str,
    value: Any,
    instance_base: str,
    verrors: List[Dict[str, Any]],
    *,
    location: Optional[str] = None,
    name: Optional[str] = None,
) -> None:
    """Validate one recorded value, mapping violations to verify errors."""
    validator = SchemaValidator(document, resolver, side, schema_pointer, issues)
    for error in validator.run(schema, value, schema_pointer):
        verrors.append(
            {
                "side": side,
                "pointer": error.schema_pointer,
                "instance_pointer": instance_base + error.instance_pointer,
                "keyword": error.keyword,
                "location": location,
                "name": name,
                "reason": _record_reason(error.reason),
            }
        )


def _record_reason(reason: str) -> str:
    """Reword shared validator messages for a recorded exchange."""
    return reason.replace("request example", "recorded request").replace(
        "response example", "recorded response"
    )


def _dedup(verrors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop exact duplicates (shared $ref'd schemas can repeat a finding)."""
    seen = set()
    unique: List[Dict[str, Any]] = []
    for error in verrors:
        key = (
            error["side"],
            error["pointer"],
            error["instance_pointer"],
            error["keyword"],
            str(error["location"]),
            str(error["name"]),
            error["reason"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(error)
    return unique


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def verify_interaction(
    document: Any,
    selector: Any,
    request_record: Any,
    response_record: Any,
) -> Dict[str, Any]:
    """Verify one recorded request/response pair against the contract.

    :raises InvalidDocumentError: *document* is not OpenAPI 3.x (400).
    :raises InvalidSelectorError: the selector is malformed/unknown (400).
    :raises InvalidVerifyRequestError: the record envelope is malformed (400).
    :raises VerificationFailedError: contract-level reference, media-type
        or response-selection problems (422); no partial result.
    """
    # --- Document shape (400) ---------------------------------------------
    shape_messages = validate_openapi_shape(document)
    if shape_messages:
        raise InvalidDocumentError(shape_messages)
    is31 = str(document.get("openapi", "")).startswith(("3.1", "3.2"))

    # --- Selector (400) -----------------------------------------------------
    parsed = parse_selector(selector)
    if parsed is None:
        raise InvalidSelectorError(
            [
                f"malformed selector {selector!r}: expected 'METHOD /path' "
                "(e.g. 'GET /pets/{petId}')"
            ]
        )
    method, path = parsed
    if path not in document["paths"]:
        raise InvalidSelectorError(
            [f"{method.upper()} {path}: path is not declared"]
        )

    # --- Record envelopes (400) ---------------------------------------------
    request, request_messages = _normalize_request_record(request_record)
    response, response_messages = _normalize_response_record(response_record)
    envelope_messages = request_messages + response_messages
    if envelope_messages:
        raise InvalidVerifyRequestError(envelope_messages)
    assert request is not None and response is not None  # messages otherwise
    status_code = response["status_code"]
    assert status_code is not None  # validated in the envelope

    # --- Path item and operation (422 references, 400 unknown method) -------
    path_pointer = pointer_join("/paths", path)
    path_item, _item_pointer, item_error = _resolve_path_item(
        document, document["paths"][path], path_pointer, is31
    )
    if item_error is not None:
        raise VerificationFailedError(_sorted_contract_errors([item_error]))

    method_keys = {
        key.lower(): key for key in path_item if key.lower() in HTTP_METHODS
    }
    if method not in method_keys:
        raise InvalidSelectorError(
            [f"{method.upper()} {path}: operation is not declared"]
        )
    operation_key = method_keys[method]
    operation = path_item[operation_key]
    operation_pointer = pointer_join(path_pointer, operation_key)
    if not isinstance(operation, dict):
        raise VerificationFailedError(
            _sorted_contract_errors(
                [
                    _contract_error(
                        "invalid_operation",
                        operation_pointer,
                        "operation must be an object",
                    )
                ]
            )
        )

    errors: List[Dict[str, Any]] = []

    # --- Effective parameters (contract level) ------------------------------
    parameters, param_errors = _build_parameters(
        document, path_item, operation, path_pointer, operation_pointer, is31
    )
    errors.extend(param_errors)

    # --- requestBody resolution ----------------------------------------------
    request_content: Optional[Dict[str, Any]] = None
    request_content_pointer: Optional[str] = None
    request_body_pointer: Optional[str] = None
    request_required = False
    raw_body = operation.get("requestBody")
    if raw_body is not None:
        request_body_pointer = pointer_join(operation_pointer, "requestBody")
        request_body, body_provenance, _terminal, body_error = (
            _resolve_referenced_object(
                document, raw_body, request_body_pointer, is31, kind="requestBody"
            )
        )
        if body_error is not None:
            errors.append(body_error)
        else:
            request_required = bool(request_body.get("required"))
            content_member_pointer = _member_pointer(
                "content", request_body_pointer, body_provenance
            )
            if "content" not in request_body:
                errors.append(
                    _contract_error(
                        "invalid_operation",
                        content_member_pointer,
                        "requestBody must declare a 'content' object",
                    )
                )
            elif not isinstance(request_body["content"], dict):
                errors.append(
                    _contract_error(
                        "invalid_operation",
                        content_member_pointer,
                        "requestBody 'content' must be an object",
                    )
                )
            else:
                request_content = request_body["content"]
                request_content_pointer = content_member_pointer
                _check_content_map(request_content, content_member_pointer, errors)

    # --- Response selection (exact -> nXX -> default) ------------------------
    responses_pointer = pointer_join(operation_pointer, "responses")
    responses = operation.get("responses")
    response_content: Optional[Dict[str, Any]] = None
    response_content_pointer: Optional[str] = None
    if not isinstance(responses, dict) or not responses:
        errors.append(
            _contract_error(
                "invalid_operation",
                responses_pointer,
                "operation must declare a non-empty 'responses' object",
            )
        )
    else:
        response_key, _match = select_response_key(responses, status_code)
        if response_key is None:
            errors.append(
                _contract_error(
                    "response_not_declared",
                    responses_pointer,
                    f"no response matches status code {status_code}: expected "
                    "the exact code, an 'nXX' wildcard key or 'default'",
                    status_code=status_code,
                )
            )
        else:
            response_ref_pointer = pointer_join(responses_pointer, response_key)
            response_obj, response_provenance, _terminal, resp_error = (
                _resolve_referenced_object(
                    document,
                    responses[response_key],
                    response_ref_pointer,
                    is31,
                    kind="response",
                )
            )
            if resp_error is not None:
                errors.append(resp_error)
            else:
                content = response_obj.get("content")
                if content is not None and not isinstance(content, dict):
                    errors.append(
                        _contract_error(
                            "invalid_operation",
                            _member_pointer(
                                "content", response_ref_pointer, response_provenance
                            ),
                            "response 'content' must be an object",
                        )
                    )
                elif isinstance(content, dict):
                    response_content = content
                    response_content_pointer = _member_pointer(
                        "content", response_ref_pointer, response_provenance
                    )
                    _check_content_map(
                        response_content, response_content_pointer, errors
                    )

    # --- Representation selection (request Content-Type / Accept) ------------
    request_media_key: Optional[str] = None
    if request_content is not None:
        if request["requested_media"] is not None:
            request_media_key = select_request_media_type(
                request_content, request["requested_media"]
            )
            if request_media_key is None:
                errors.append(
                    _contract_error(
                        "unsupported_media_type",
                        request_content_pointer,
                        f"the recorded request Content-Type "
                        f"{request['content_type']!r} matches none of the "
                        "declared requestBody media types",
                        supported=sorted(request_content),
                    )
                )
        elif request["body_present"]:
            errors.append(
                _contract_error(
                    "unsupported_media_type",
                    request_content_pointer,
                    "the recorded request carries a body but no Content-Type "
                    "to select a requestBody representation with",
                    supported=sorted(request_content),
                )
            )

    response_media_key: Optional[str] = None
    if response_content is not None:
        response_media_key = select_response_media_type(
            response_content, request["accept_ranges"]
        )
        if response_media_key is None:
            errors.append(
                _contract_error(
                    "not_acceptable",
                    response_content_pointer,
                    f"none of the declared response media types is acceptable "
                    f"under the recorded Accept header {request['accept']!r}",
                    available=sorted(response_content),
                )
            )

    if errors:
        raise VerificationFailedError(_sorted_contract_errors(errors))

    # ------------------------------------------------------------------ #
    # Validation phase (everything above resolved successfully)           #
    # ------------------------------------------------------------------ #
    resolver = RefResolver(document, label="document")
    issues = _RefIssues()
    verrors: List[Dict[str, Any]] = []

    # Parameters: required presence, then schema type/enum/readOnly rules.
    for param in parameters:
        found, value = _lookup_value(
            request[param.location], param.location, param.name
        )
        instance_base = pointer_join(
            pointer_join("/request", param.location), param.name
        )
        if not found:
            if param.required:
                verrors.append(
                    {
                        "side": "request",
                        "pointer": param.pointer,
                        "instance_pointer": pointer_join(
                            "/request", param.location
                        ),
                        "keyword": "required",
                        "location": param.location,
                        "name": param.name,
                        "reason": (
                            f"missing required {param.location} parameter "
                            f"'{param.name}'"
                        ),
                    }
                )
            continue
        schema = param.node.get("schema")
        if schema is None:
            continue
        _run_schema_validation(
            document,
            resolver,
            issues,
            "request",
            schema,
            param.pointer + "/schema",
            value,
            instance_base,
            verrors,
            location=param.location,
            name=param.name,
        )

    # Request body against the selected requestBody representation.
    if request_media_key is not None and request["body_present"]:
        media_node = request_content[request_media_key]
        schema = media_node.get("schema")
        if schema is not None:
            media_pointer = pointer_join(
                request_content_pointer, request_media_key
            )
            _run_schema_validation(
                document,
                resolver,
                issues,
                "request",
                schema,
                media_pointer + "/schema",
                request["body"],
                "/request/body",
                verrors,
            )
    if request_required and not request["body_present"]:
        verrors.append(
            {
                "side": "request",
                "pointer": request_body_pointer,
                "instance_pointer": "/request/body",
                "keyword": "required",
                "location": None,
                "name": None,
                "reason": (
                    "the operation declares a required request body but the "
                    "recorded request carries none"
                ),
            }
        )

    # Response body against the representation selected for Accept. A
    # response with declared content must carry a conforming body, so an
    # absent recorded body is validated as JSON null.
    if response_media_key is not None:
        media_node = response_content[response_media_key]
        schema = media_node.get("schema")
        if schema is not None:
            media_pointer = pointer_join(
                response_content_pointer, response_media_key
            )
            _run_schema_validation(
                document,
                resolver,
                issues,
                "response",
                schema,
                media_pointer + "/schema",
                response["body"] if response["body_present"] else None,
                "/response/body",
                verrors,
            )

    # Reference problems reached while validating are 422-class: report
    # them together and discard every partial validation result.
    try:
        issues.raise_if_any()
    except ExampleReferenceError as exc:
        raise VerificationFailedError(
            _sorted_contract_errors(
                [
                    {
                        "code": exc.code,
                        "pointer": reference["pointer"],
                        "ref": reference["ref"],
                        "error": reference["error"],
                    }
                    for reference in exc.references
                ]
            )
        ) from None

    unique = _dedup(verrors)
    # Stable order: source JSON Pointer first, then the side, the location
    # inside the record, the violated keyword and the reason.
    unique.sort(
        key=lambda error: (
            error["pointer"],
            error["side"],
            error["instance_pointer"],
            error["keyword"],
            error["reason"],
        )
    )
    return {
        "valid": not unique,
        "error_count": len(unique),
        "errors": unique,
    }
