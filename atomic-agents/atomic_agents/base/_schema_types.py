"""Resolve schema parameters through reusable generic subclasses."""

from typing import Any, get_args, get_origin

from atomic_agents.base.base_io_schema import BaseIOSchema


def resolve_schema_types(instance: object, schema_base: type) -> tuple[type[BaseIOSchema], type[BaseIOSchema]] | None:
    """Find the first schema binding in left-to-right declared-base order.

    An inherited specialization takes precedence over a later redeclared base.
    Callers supply their own defaults when no schema-base binding is found.
    """

    def resolve(base: Any, bindings: dict[Any, Any]) -> tuple[Any, Any] | None:
        origin = get_origin(base) or base
        arguments = tuple(bindings.get(argument, argument) for argument in get_args(base))
        if origin is schema_base:
            return (arguments[0], arguments[1]) if len(arguments) == 2 else None

        bindings = {**bindings, **dict(zip(getattr(origin, "__parameters__", ()), arguments))}
        # Inherited __orig_bases__ can skip an intermediate specialization.
        for parent in origin.__dict__.get("__orig_bases__", origin.__bases__):
            schemas = resolve(parent, bindings)
            if schemas is not None:
                return schemas
        return None

    return resolve(getattr(instance, "__orig_class__", type(instance)), {})
