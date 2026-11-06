"""Resolve schema parameters through reusable component subclasses."""

from typing import Any, get_args, get_origin

from atomic_agents.base.base_io_schema import BaseIOSchema


def resolve_schema_types(instance: Any, schema_base: type) -> tuple[type[BaseIOSchema], type[BaseIOSchema]]:
    """Resolve a component's input/output types without caching unbound parameters."""

    def resolve(base: Any, bindings: dict[Any, Any]) -> tuple[Any, Any] | None:
        origin = get_origin(base) or base
        arguments = tuple(bindings.get(argument, argument) for argument in get_args(base))
        if origin is schema_base:
            return (arguments[0], arguments[1]) if len(arguments) == 2 else None

        bindings = {**bindings, **dict(zip(getattr(origin, "__parameters__", ()), arguments))}
        # Inherited __orig_bases__ can skip an intermediate specialization.
        parents = origin.__dict__.get("__orig_bases__", origin.__bases__)
        # An explicitly redeclared component base overrides inherited schemas.
        for parent in parents:
            if get_origin(parent) is schema_base:
                return resolve(parent, bindings)
        for parent in parents:
            schemas = resolve(parent, bindings)
            if schemas is not None:
                return schemas
        return None

    instance_type = getattr(instance, "__orig_class__", type(instance))
    schemas = resolve(instance_type, {})
    if schemas is not None:
        return schemas

    # Preserve legacy generic subclasses that do not forward parameters to the base.
    arguments = get_args(instance_type)
    if len(arguments) == 2:
        return arguments[0], arguments[1]
    return BaseIOSchema, BaseIOSchema
