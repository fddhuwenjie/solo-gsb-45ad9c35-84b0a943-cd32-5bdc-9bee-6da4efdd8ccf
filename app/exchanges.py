"""Contract verification of one recorded joint-debugging exchange.

``POST /exchanges/verify`` receives an OpenAPI 3.x document, one
``METHOD /path`` operation selector and a *real* HTTP exchange:

* request: ``path`` / ``query`` / ``header`` / ``cookie`` parameter
  values, a ``Content-Type`` / ``Accept`` header and a JSON body;
* response: a status code, headers, a ``Content-Type`` and a body.

It then checks the exchange against the contract:

* path-level parameters are merged with operation-level ones, the
  operation-level entry overriding the path-level one by ``(in, name)``;
* every parameter is checked for presence (required), JSON type and enum,
  with ``readOnly`` forbidden on the request side and ``writeOnly``
  forbidden on the response side;
* the response is selected by exact status code, ``nXX`` wildcard and
  ``default`` (in that order);
* the request representation is chosen from the request Content-Type and
  the response representation from the actual response Content-Type
  (falling back to Accept negotiation when the response omits it);
* the JSON bodies on both sides are validated against their media type
  schema, including direction-aware ``required`` / ``readOnly`` /
  ``writeOnly`` semantics.

Reference / media / response-selection problems are contract failures and
reject the whole request with HTTP 422 (no partial result); a body that
simply violates its schema is reported as a violation in a ``200``
response with ``valid = false``. Violations are sorted stably by their
source JSON Pointer. The input is never modified.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .examples import SchemaValidator, _RefIssues, _is_json_media_type
from .materializer import (
    GROUPS,
    Parameter,
    _build_parameters,
    _resolve_path_item as _resolve_parameter_path_item,
    pointer_join,
)
from .negotiation import (
    MediaTypeParseError,
    NegotiationError,
    _contract_error,
    _member_pointer,
    _parse_declared_media_type,
    _resolve_referenced_object,
    _sorted_errors,
    canonical_media_type,
    normalize_status_code,
    parse_accept,
    parse_media_type,
    quality_for,
    select_request_media_type,
    select_response_key,
    select_response_media_type,
)
from .refresolver import RefResolver
from .slicer import HTTP_METHODS, parse_selector, validate_openapi_shape

REQUEST_SIDE = "request"
RESPONSE_SIDE = "response"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ExchangeError(NegotiationError):
    """Base class for exchange verification failures."""


class InvalidDocumentError(ExchangeError):
    """The submitted document is not a recognizable OpenAPI 3.x doc (400)."""

    def __init__(self, messages: List[str]) -> None:
        self.messages = messages
        super().__init__("; ".join(messages))


class InvalidSelectorError(ExchangeError):
    """The selector is malformed or unknown (400)."""

    def __init__(self, messages: List[str]) -> None:
        self.messages = messages
        super().__init__("; ".join(messages))


class InvalidExchangeRequestError(ExchangeError):
    """The exchange envelope (parameters, headers, body) is malformed (400)."""

    def __init__(self, messages: List[str]) -> None:
        self.messages = messages
        super().__init__("; ".join(messages))


class ExchangeContractError(ExchangeError):
    """A contract-level failure (reference / media / selection) (422)."""

    def __init__(self, errors: List[Dict[str, Any]]) -> None:
        self.errors = _sorted_errors(errors)
        super().__init__(
            "; ".join(f"{e['code']} @ {e.get('pointer', '')}" for e in self.errors)
        )


# --------------------------------------------------------------------------- #
# Envelope validation
# --------------------------------------------------------------------------- #


def _validate_parameter_group(
    request: Dict[str, Any], group: str, messages: List[str]
) -> Dict[str, Any]:
    """Extract one parameter group, validating its envelope shape."""
    value = request.get(group)
    if value is None:
        return {}
    if not isinstance(value, dict):
        messages.append(
            f"request.{group} must be an object mapping parameter names to JSON values"
        )
        return {}
    if any(not isinstance(key, str) for key in value):
        messages.append(f"request.{group} keys must be parameter name strings")
    return value


def _validate_header_envelope(
    container: Any, path: str, messages: List[str]
) -> Dict[str, Any]:
    headers = container.get("headers")
    if headers is None:
        return {}
    if not isinstance(headers, dict):
        messages.append(f"{path} must be an object mapping header names to JSON values")
        return {}
    if any(not isinstance(key, str) for key in headers):
        messages.append(f"{path} keys must be header name strings")
    return headers


def _validate_exchange(
    exchange: Any,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    int,
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Validate the exchange envelope; return its normalized pieces.

    Returns ``(request, response, code, request_media, response_media,
    accept_ranges)``.
    """
    if not isinstance(exchange, dict):
        raise InvalidExchangeRequestError(["exchange must be a JSON object"])

    messages: List[str] = []

    request = exchange.get("request")
    if request is None:
        request = {}
    elif not isinstance(request, dict):
        messages.append("exchange.request must be an object")
        request = {}

    response = exchange.get("response")
    if not isinstance(response, dict):
        messages.append("exchange.response must be an object")
        response = {}

    code, code_error = normalize_status_code(response.get("status_code"))
    if code_error is not None:
        messages.append(code_error)

    request_media = _parse_concrete_media(
        request.get("content_type"), "request.content_type", messages
    )
    response_media = _parse_concrete_media(
        response.get("content_type"), "response.content_type", messages
    )

    accept_ranges: List[Dict[str, Any]] = [
        {"type": "*", "subtype": "*", "parameters": {}, "q": 1.0}
    ]
    raw_accept = request.get("accept")
    if raw_accept is not None:
        if not isinstance(raw_accept, str):
            messages.append("request.accept must be a string when provided")
        else:
            accept_ranges, accept_errors = parse_accept(raw_accept)
            messages.extend(accept_errors)

    for group in GROUPS:
        _validate_parameter_group(request, group, messages)
    _validate_header_envelope(response, "response.headers", messages)

    # Ambiguous on-the-wire headers: two JSON keys differing only in case.
    for label, container in (
        ("request.header", request.get("header")),
        ("response.headers", response.get("headers")),
    ):
        if isinstance(container, dict):
            seen: Dict[str, str] = {}
            for key in container:
                if not isinstance(key, str):
                    continue
                if key.lower() in seen and seen[key.lower()] != key:
                    messages.append(
                        f"{label} keys {seen[key.lower()]!r} and {key!r} differ "
                        "only in case; supply one header name"
                    )
                seen[key.lower()] = key

    if messages:
        raise InvalidExchangeRequestError(messages)
    assert code is not None
    return request, response, code, request_media, response_media, accept_ranges


def _parse_concrete_media(
    raw: Any, field: str, messages: List[str]
) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        messages.append(f"{field} must be a string when provided")
        return None
    try:
        return parse_media_type(raw)
    except MediaTypeParseError as exc:
        messages.append(str(exc))
        return None


# --------------------------------------------------------------------------- #
# Contract object resolution
# --------------------------------------------------------------------------- #


def _validate_content_map(content: Dict[str, Any], content_pointer: str) -> None:
    """Reject malformed content-map keys / media type objects (422)."""
    errors: List[Dict[str, Any]] = []
    for key, node in content.items():
        if not isinstance(key, str):
            errors.append(
                _contract_error(
                    "invalid_media_type",
                    content_pointer,
                    "content keys must be media type strings",
                )
            )
            continue
        media_pointer = pointer_join(content_pointer, key)
        try:
            _parse_declared_media_type(key)
        except MediaTypeParseError as exc:
            errors.append(
                _contract_error("invalid_media_type", media_pointer, str(exc))
            )
        if not isinstance(node, dict):
            errors.append(
                _contract_error(
                    "invalid_operation",
                    media_pointer,
                    f"media type object for {key!r} must be an object",
                )
            )
    if errors:
        raise ExchangeContractError(errors)


def _resolve_request_body(
    doc: Dict[str, Any],
    operation: Dict[str, Any],
    operation_pointer: str,
    is31: bool,
) -> Optional[Tuple[Dict[str, Any], str, str]]:
    """Resolve the operation requestBody and validate its content object.

    Returns ``(body, body_occurrence_pointer, content_pointer)`` or
    ``None`` when no requestBody is declared.
    """
    raw_body = operation.get("requestBody")
    if raw_body is None:
        return None
    body_pointer = pointer_join(operation_pointer, "requestBody")
    body, provenance, _terminal, error = _resolve_referenced_object(
        doc, raw_body, body_pointer, is31, kind="requestBody"
    )
    if error is not None:
        raise ExchangeContractError([error])
    content_pointer = _member_pointer("content", body_pointer, provenance)
    if "content" not in body:
        raise ExchangeContractError(
            [
                _contract_error(
                    "invalid_operation",
                    content_pointer,
                    "requestBody must declare a 'content' object",
                )
            ]
        )
    if not isinstance(body["content"], dict):
        raise ExchangeContractError(
            [
                _contract_error(
                    "invalid_operation",
                    content_pointer,
                    "requestBody 'content' must be an object",
                )
            ]
        )
    _validate_content_map(body["content"], content_pointer)
    return body, body_pointer, content_pointer


def _resolve_response(
    doc: Dict[str, Any],
    operation: Dict[str, Any],
    operation_pointer: str,
    code: int,
    is31: bool,
) -> Tuple[Dict[str, Any], str, str, str, str, Optional[Dict[str, str]]]:
    """Select (exact -> nXX -> default) and resolve the response for *code*.

    Returns ``(response, response_pointer, response_key, match,
    content_pointer, response_provenance)``; *content_pointer* is
    ``""`` when the response declares no content.
    """
    responses_pointer = pointer_join(operation_pointer, "responses")
    responses = operation.get("responses")
    if not isinstance(responses, dict) or not responses:
        raise ExchangeContractError(
            [
                _contract_error(
                    "invalid_operation",
                    responses_pointer,
                    "operation must declare a non-empty 'responses' object",
                )
            ]
        )
    response_key, match = select_response_key(responses, code)
    if response_key is None:
        raise ExchangeContractError(
            [
                _contract_error(
                    "response_not_declared",
                    responses_pointer,
                    f"no response matches status code {code}: expected the "
                    "exact code, an 'nXX' wildcard key or 'default'",
                    status_code=code,
                )
            ]
        )
    response_pointer = pointer_join(responses_pointer, response_key)
    response, provenance, _terminal, error = _resolve_referenced_object(
        doc,
        responses[response_key],
        response_pointer,
        is31,
        kind="response",
    )
    if error is not None:
        raise ExchangeContractError([error])
    content_pointer = ""
    if response.get("content") is not None:
        content_pointer = _member_pointer("content", response_pointer, provenance)
        if not isinstance(response["content"], dict):
            raise ExchangeContractError(
                [
                    _contract_error(
                        "invalid_operation",
                        content_pointer,
                        "response 'content' must be an object",
                    )
                ]
            )
        _validate_content_map(response["content"], content_pointer)
    return response, response_pointer, response_key, match or "", content_pointer, provenance


# --------------------------------------------------------------------------- #
# Representation selection
# --------------------------------------------------------------------------- #


def _canonical(media: Dict[str, Any]) -> str:
    return canonical_media_type(
        media["type"], media["subtype"], media["parameters"]
    )


def _is_declared_json(declared_key: str) -> bool:
    parsed = _parse_declared_media_type(declared_key)
    return _is_json_media_type(
        canonical_media_type(parsed["type"], parsed["subtype"], parsed["parameters"])
    )


def _select_request_representation(
    body: Dict[str, Any],
    body_pointer: str,
    content: Dict[str, Any],
    content_pointer: str,
    actual_media: Optional[Dict[str, Any]],
    has_body: bool,
) -> Optional[Tuple[Dict[str, Any], str, str]]:
    """Pick the request media type from the actual Content-Type.

    Returns ``(media_node, declared_key, media_pointer)`` or ``None``
    when no request body was recorded. A required-but-missing body, a
    body sent without a Content-Type, or an unmatched Content-Type is a
    contract failure (422).
    """
    required = bool(body.get("required"))
    if not has_body:
        if required:
            raise ExchangeContractError(
                [
                    _contract_error(
                        "request_body_missing",
                        body_pointer,
                        "the requestBody is marked required but the recorded "
                        "request carries no body",
                    )
                ]
            )
        return None
    if actual_media is None:
        raise ExchangeContractError(
            [
                _contract_error(
                    "request_body_content_type_missing",
                    body_pointer,
                    "the recorded request carries a body without a Content-Type; "
                    "a request representation cannot be selected",
                )
            ]
        )
    chosen_key = select_request_media_type(content, actual_media)
    if chosen_key is None:
        raise ExchangeContractError(
            [
                _contract_error(
                    "unsupported_media_type",
                    content_pointer,
                    "the request Content-Type does not match any requestBody "
                    "representation declared by the operation",
                    content_type=_canonical(actual_media),
                    supported=list(content),
                )
            ]
        )
    return content[chosen_key], chosen_key, pointer_join(content_pointer, chosen_key)


def _select_response_representation(
    content: Dict[str, Any],
    content_pointer: str,
    actual_media: Optional[Dict[str, Any]],
    accept_ranges: List[Dict[str, Any]],
    has_body: bool,
) -> Optional[Tuple[Dict[str, Any], str, str]]:
    """Pick the response media type.

    The actual response Content-Type wins when present (it must match a
    declared type and be acceptable under Accept); otherwise the
    representation is selected purely through Accept negotiation. A
    content-bearing response for which no representation can be
    identified is a contract failure (422).
    """
    if not has_body:
        return None

    if actual_media is not None:
        chosen_key = select_request_media_type(content, actual_media)
        if chosen_key is None:
            raise ExchangeContractError(
                [
                    _contract_error(
                        "response_media_type_not_declared",
                        content_pointer,
                        "the response Content-Type does not match any "
                        "representation declared for the selected response",
                        content_type=_canonical(actual_media),
                        available=list(content),
                    )
                ]
            )
        qvalue = quality_for(actual_media, accept_ranges)
        if qvalue is None or qvalue <= 0.0:
            raise ExchangeContractError(
                [
                    _contract_error(
                        "not_acceptable",
                        content_pointer,
                        "the response Content-Type is not acceptable under the "
                        "request Accept header",
                        content_type=_canonical(actual_media),
                    )
                ]
            )
    else:
        chosen_key = select_response_media_type(content, accept_ranges)
        if chosen_key is None:
            raise ExchangeContractError(
                [
                    _contract_error(
                        "response_media_type_missing",
                        content_pointer,
                        "the recorded response carries a body without a "
                        "Content-Type and no representation can be selected "
                        "from the Accept header (missing, empty or q=0)",
                        available=list(content),
                    )
                ]
            )
    return content[chosen_key], chosen_key, pointer_join(content_pointer, chosen_key)


# --------------------------------------------------------------------------- #
# Violations
# --------------------------------------------------------------------------- #


def _violation(
    *,
    side: str,
    source_pointer: str,
    instance_pointer: str,
    keyword: str,
    reason: str,
    location: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "side": side,
        "location": location,
        "name": name,
        "source_pointer": source_pointer,
        "instance_pointer": instance_pointer,
        "keyword": keyword,
        "reason": reason,
    }


def _from_schema_error(error: Any, location: str, name: str) -> Dict[str, Any]:
    return _violation(
        side=error.side,
        location=location,
        name=name,
        source_pointer=error.schema_pointer,
        instance_pointer=error.instance_pointer or "",
        keyword=error.keyword,
        reason=error.reason,
    )


def _root_direction_marker(schema: Any, side: str) -> Optional[str]:
    """The violated direction marker on a root (parameter/header) schema."""
    if not isinstance(schema, dict):
        return None
    if side == REQUEST_SIDE and schema.get("readOnly") is True:
        return "readOnly"
    if side == RESPONSE_SIDE and schema.get("writeOnly") is True:
        return "writeOnly"
    return None


def _validate_param_value(
    doc: Dict[str, Any],
    resolver: RefResolver,
    issues: _RefIssues,
    location: str,
    name: str,
    node: Dict[str, Any],
    definition_pointer: str,
    value: Any,
    side: str,
    provenance: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Validate one concrete parameter/header value against ``schema``."""
    if "schema" not in node:
        # A 'content'-based parameter/header places no directly evaluable
        # JSON schema on the value; presence is enforced by the caller.
        return []
    schema_pointer = _member_pointer("schema", definition_pointer, provenance)
    schema = node["schema"]
    target, physical = issues.follow(resolver, schema, schema_pointer)
    marker = _root_direction_marker(target, side)
    if marker is not None:
        return [
            _violation(
                side=side,
                location=location,
                name=name,
                source_pointer=physical or schema_pointer,
                instance_pointer="",
                keyword=marker,
                reason=(
                    f"{marker} {location} '{name}' must not be sent on a {side}"
                ),
            )
        ]
    validator = SchemaValidator(doc, resolver, side, "", issues)
    raw_errors = validator.run(schema, value, schema_pointer)
    return [_from_schema_error(e, location, name) for e in raw_errors]


def _validate_body(
    doc: Dict[str, Any],
    resolver: RefResolver,
    issues: _RefIssues,
    media_node: Dict[str, Any],
    media_pointer: str,
    body: Any,
    side: str,
) -> List[Dict[str, Any]]:
    schema = media_node.get("schema")
    if schema is None:
        return []
    schema_pointer = media_pointer + "/schema"
    validator = SchemaValidator(doc, resolver, side, "", issues)
    raw_errors = validator.run(schema, body, schema_pointer)
    return [
        _violation(
            side=e.side,
            source_pointer=e.schema_pointer,
            instance_pointer=e.instance_pointer or "",
            keyword=e.keyword,
            reason=e.reason,
        )
        for e in raw_errors
    ]


def _sort_violations(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable ordering by side, then source JSON Pointer."""
    return sorted(
        errors,
        key=lambda e: (
            str(e.get("side", "")),
            str(e.get("source_pointer", "")),
            str(e.get("instance_pointer", "")),
            str(e.get("location") or ""),
            str(e.get("name") or ""),
            str(e.get("keyword", "")),
            str(e.get("reason", "")),
        ),
    )


# --------------------------------------------------------------------------- #
# Request parameters / response headers
# --------------------------------------------------------------------------- #


def _header_lookup(values: Dict[str, Any], name: str) -> Tuple[bool, Any]:
    """Case-insensitive header lookup in a JSON name -> value map."""
    lowered = name.lower()
    for key, value in values.items():
        if isinstance(key, str) and key.lower() == lowered:
            return True, value
    return False, None


def _check_request_parameters(
    doc: Dict[str, Any],
    resolver: RefResolver,
    issues: _RefIssues,
    parameters: List[Parameter],
    groups: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    for param in parameters:
        supplied = groups.get(param.location, {})
        if param.location == "header":
            found, value = _header_lookup(supplied, param.name)
        else:
            found, value = param.name in supplied, supplied.get(param.name)

        # Missing or an explicit JSON null means "no value" on the wire.
        if not found or value is None:
            if param.required:
                violations.append(
                    _violation(
                        side=REQUEST_SIDE,
                        location=param.location,
                        name=param.name,
                        source_pointer=param.pointer,
                        instance_pointer="",
                        keyword="required",
                        reason=(
                            f"missing value for required {param.location} "
                            f"parameter '{param.name}'"
                        ),
                    )
                )
            continue
        violations.extend(
            _validate_param_value(
                doc,
                resolver,
                issues,
                param.location,
                param.name,
                param.node,
                param.pointer,
                value,
                REQUEST_SIDE,
            )
        )
    return violations


def _check_response_headers(
    doc: Dict[str, Any],
    resolver: RefResolver,
    issues: _RefIssues,
    response: Dict[str, Any],
    response_pointer: str,
    response_provenance: Optional[Dict[str, str]],
    headers: Dict[str, Any],
    is31: bool,
) -> List[Dict[str, Any]]:
    declared = response.get("headers")
    violations: List[Dict[str, Any]] = []
    if not isinstance(declared, dict) or not declared:
        return violations
    headers_base = _member_pointer("headers", response_pointer, response_provenance)
    for name, raw_header in declared.items():
        if not isinstance(name, str):
            continue
        header_pointer = pointer_join(headers_base, name)
        header, header_provenance, _terminal, error = _resolve_referenced_object(
            doc, raw_header, header_pointer, is31, kind="header"
        )
        if error is not None:
            raise ExchangeContractError([error])
        if not isinstance(header, dict):
            raise ExchangeContractError(
                [
                    _contract_error(
                        "invalid_operation",
                        header_pointer,
                        f"header {name!r} must resolve to an object",
                    )
                ]
            )
        found, value = _header_lookup(headers, name)
        if not found or value is None:
            if bool(header.get("required")):
                violations.append(
                    _violation(
                        side=RESPONSE_SIDE,
                        location="header",
                        name=name,
                        source_pointer=header_pointer,
                        instance_pointer="",
                        keyword="required",
                        reason=f"missing value for required response header '{name}'",
                    )
                )
            continue
        violations.extend(
            _validate_param_value(
                doc,
                resolver,
                issues,
                "header",
                name,
                header,
                header_pointer,
                value,
                RESPONSE_SIDE,
                header_provenance,
            )
        )
    return violations


# --------------------------------------------------------------------------- #
# Reference issues
# --------------------------------------------------------------------------- #


def _raise_reference_issues(issues: _RefIssues) -> None:
    """Turn collected schema reference problems into a 422."""
    errors: List[Dict[str, Any]] = []
    for bucket, code in (
        (issues.external, "unsupported_reference"),
        (issues.invalid, "invalid_reference"),
        (issues.dangling, "unresolvable_reference"),
    ):
        for key in sorted(bucket):
            entry = dict(bucket[key])
            entry["code"] = code
            errors.append(entry)
    if errors:
        raise ExchangeContractError(errors)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def verify_exchange(doc: Any, selector: Any, exchange: Any) -> Dict[str, Any]:
    """Verify one recorded HTTP exchange against the contract.

    :raises InvalidDocumentError: *doc* is not OpenAPI 3.x (400).
    :raises InvalidSelectorError: the selector is malformed/unknown (400).
    :raises InvalidExchangeRequestError: malformed exchange envelope (400).
    :raises ExchangeContractError: reference / media / selection failure
        (422); no partial verdict is returned.
    """
    shape_messages = validate_openapi_shape(doc)
    if shape_messages:
        raise InvalidDocumentError(shape_messages)

    parsed = parse_selector(selector)
    if parsed is None:
        raise InvalidSelectorError(
            [
                f"malformed selector {selector!r}: expected 'METHOD /path' "
                "(e.g. 'POST /pets')"
            ]
        )
    method, path = parsed
    if path not in doc["paths"]:
        raise InvalidSelectorError([f"{method.upper()} {path}: path is not declared"])

    request, response, code, request_media, response_media, accept_ranges = (
        _validate_exchange(exchange)
    )

    is31 = str(doc.get("openapi", "")).startswith(("3.1", "3.2"))
    resolver = RefResolver(doc)
    issues = _RefIssues()

    # --- Operation location ------------------------------------------------
    path_pointer = pointer_join("/paths", path)
    path_item, item_error = _resolve_parameter_path_item(
        doc, doc["paths"][path], path_pointer
    )
    if item_error is not None:
        raise ExchangeContractError([item_error])
    if not isinstance(path_item, dict):
        raise ExchangeContractError(
            [
                _contract_error(
                    "invalid_operation", path_pointer, "path item must be an object"
                )
            ]
        )
    method_keys = {k.lower(): k for k in path_item if k.lower() in HTTP_METHODS}
    if method not in method_keys:
        raise InvalidSelectorError(
            [f"{method.upper()} {path}: operation is not declared"]
        )
    method_key = method_keys[method]
    operation = path_item[method_key]
    operation_pointer = pointer_join(path_pointer, method_key)
    if not isinstance(operation, dict):
        raise ExchangeContractError(
            [
                _contract_error(
                    "invalid_operation",
                    operation_pointer,
                    "operation must be an object",
                )
            ]
        )

    # Envelope validation already ensured each group is an object when
    # present; a non-object group raised 400 before this point.
    groups = {group: request.get(group) or {} for group in GROUPS}
    response_headers = response.get("headers") or {}

    # --- Parameters (path/operation merge) ----------------------------------
    parameters, parameter_errors = _build_parameters(
        doc, path_item, operation, path_pointer, operation_pointer, is31
    )
    if parameter_errors:
        raise ExchangeContractError(parameter_errors)

    # --- Request body representation ---------------------------------------
    resolved_body = _resolve_request_body(doc, operation, operation_pointer, is31)
    request_selection: Optional[Tuple[Dict[str, Any], str, str]] = None
    request_body = request.get("body")
    request_has_body = request_media is not None or request_body is not None
    if resolved_body is not None:
        body_node, body_pointer, content_pointer = resolved_body
        request_selection = _select_request_representation(
            body_node,
            body_pointer,
            body_node["content"],
            content_pointer,
            request_media,
            request_has_body,
        )

    # --- Response selection and representation ------------------------------
    resp_node, resp_pointer, resp_key, resp_match, resp_content_pointer, resp_prov = (
        _resolve_response(doc, operation, operation_pointer, code, is31)
    )
    response_selection: Optional[Tuple[Dict[str, Any], str, str]] = None
    response_body = response.get("body")
    response_has_body = response_media is not None or response_body is not None
    if resp_content_pointer != "":
        response_selection = _select_response_representation(
            resp_node["content"],
            resp_content_pointer,
            response_media,
            accept_ranges,
            response_has_body,
        )

    # --- Violations ---------------------------------------------------------
    violations = _check_request_parameters(doc, resolver, issues, parameters, groups)
    violations.extend(
        _check_response_headers(
            doc,
            resolver,
            issues,
            resp_node,
            resp_pointer,
            resp_prov,
            response_headers,
            is31,
        )
    )

    if request_selection is not None and request_body is not None:
        media_node, declared_key, media_pointer = request_selection
        if _is_declared_json(declared_key):
            violations.extend(
                _validate_body(
                    doc, resolver, issues,
                    media_node, media_pointer, request_body, REQUEST_SIDE,
                )
            )

    if response_selection is not None and response_body is not None:
        media_node, declared_key, media_pointer = response_selection
        if _is_declared_json(declared_key):
            violations.extend(
                _validate_body(
                    doc, resolver, issues,
                    media_node, media_pointer, response_body, RESPONSE_SIDE,
                )
            )

    # Any reference problem while resolving schemas rejects the whole
    # exchange with 422 -- no half-finished verdict.
    _raise_reference_issues(issues)

    violations = _sort_violations(violations)
    return {
        "operation_pointer": operation_pointer,
        "request": _selection_summary(request_selection, request_has_body),
        "response": {
            "status_code": code,
            "key": resp_key,
            "match": resp_match,
            "response_pointer": resp_pointer,
            **_selection_summary(response_selection, response_has_body),
        },
        "valid": not violations,
        "error_count": len(violations),
        "errors": violations,
    }


def _selection_summary(
    selection: Optional[Tuple[Dict[str, Any], str, str]], has_body: bool
) -> Dict[str, Any]:
    if selection is None:
        return {
            "present": has_body,
            "declared_media_type": None,
            "media_type_pointer": None,
            "json": False,
        }
    _node, declared_key, media_pointer = selection
    return {
        "present": True,
        "declared_media_type": declared_key,
        "media_type_pointer": media_pointer,
        "json": _is_declared_json(declared_key),
    }
