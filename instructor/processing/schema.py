"""
Standalone schema generation utilities for different LLM providers.

This module provides provider-agnostic functions to generate schemas from Pydantic models
without requiring inheritance from OpenAISchema or use of decorators.
"""

from __future__ import annotations

import functools
import inspect
import re
import sys
import warnings
from enum import Enum
from typing import Any, Dict, Type

from docstring_parser import parse
from pydantic import BaseModel

from ..providers.gemini.utils import map_to_gemini_function_schema

__all__ = [
    "generate_openai_schema",
    "generate_anthropic_schema",
    "generate_gemini_schema",
]


def extract_enum_annotations(enum_class: Type[Enum]) -> Dict[str, str]:
    """
    Extract annotations (comments) from enum values.
    
    Args:
        enum_class: The enum class to extract annotations from
        
    Returns:
        A dictionary mapping enum values to their annotations
    """
    annotations = {}
    
    # Get the source code of the enum class
    try:
        source = inspect.getsource(enum_class)
        
        # Parse the source code to extract comments
        for line in source.split('\n'):
            # Look for lines with enum values and comments
            match = re.search(r'(\w+)\s*=\s*["\']([^"\']+)["\'](?:\s*#\s*(.+))?', line)
            if match:
                enum_name, enum_value, comment = match.groups()
                if comment:
                    annotations[enum_value] = comment
    except (OSError, TypeError):
        # If we can't get the source code, return empty annotations
        pass
    
    return annotations

@functools.lru_cache(maxsize=256)
def generate_openai_schema(model: type[BaseModel]) -> dict[str, Any]:
    """
    Generate OpenAI function schema from a Pydantic model.

    Args:
        model: A Pydantic BaseModel subclass

    Returns:
        A dictionary in the format of OpenAI's function schema

    Note:
        The model's docstring will be used for the function description.
        Parameter descriptions from the docstring will enrich field descriptions.
        Enum annotations (comments) will be included in the schema.
    """
    schema = model.model_json_schema()
    docstring = parse(model.__doc__ or "")
    parameters = {k: v for k, v in schema.items() if k not in ("title", "description")}

    # Enrich parameter descriptions from docstring
    for param in docstring.params:
        if (name := param.arg_name) in parameters["properties"] and (
            description := param.description
        ):
            if "description" not in parameters["properties"][name]:
                parameters["properties"][name]["description"] = description

    # Dictionary to store enum annotations for later use with properties
    enum_annotations_by_type = {}
    
    # Process enum types in $defs section
    if '$defs' in schema:
        for type_name, type_schema in schema['$defs'].items():
            if 'enum' in type_schema:
                enum_class = None
                            
                # Try multiple strategies to find the enum class
                
                # Strategy 1: Try to find the enum class in the model's field annotations
                try:
                    type_hints = inspect.get_annotations(model)
                    for field_name, field_type in type_hints.items():
                        # Check if this field is the enum we're looking for
                        if hasattr(field_type, "__name__") and field_type.__name__ == type_name:
                            if isinstance(field_type, type) and issubclass(field_type, Enum):
                                enum_class = field_type
                                break
                        # Handle the case where field_type is a generic type (like Optional[EnumType])
                        elif hasattr(field_type, "__origin__") and hasattr(field_type, "__args__"):
                            for arg in field_type.__args__:
                                if hasattr(arg, "__name__") and arg.__name__ == type_name:
                                    if isinstance(arg, type) and issubclass(arg, Enum):
                                        enum_class = arg
                                        break
                except (TypeError, AttributeError) as e:
                    pass

                # Strategy 2: Try to find the enum class in the model's module using __module__
                if not enum_class:
                    try:
                        module_name = model.__module__
                        if module_name in sys.modules:
                            module = sys.modules[module_name]
                            enum_class = getattr(module, type_name, None)
                            if enum_class and isinstance(enum_class, type) and issubclass(enum_class, Enum):
                                pass
                    except (AttributeError, KeyError) as e:
                        pass
                
                # Strategy 3: Try to find the enum class in imported modules
                if not enum_class:
                    try:
                        # Check all loaded modules for the enum class
                        for module_name, module in list(sys.modules.items()):
                            if module and not module_name.startswith('_'):
                                try:
                                    potential_enum = getattr(module, type_name, None)
                                    if (potential_enum and isinstance(potential_enum, type) and
                                        issubclass(potential_enum, Enum)):
                                        enum_class = potential_enum
                                        break
                                except (AttributeError, TypeError):
                                    continue
                    except Exception as e:
                        pass
                
                if not enum_class:
                    pass

                # If enum class is found, extract annotations
                if enum_class and isinstance(enum_class, type) and issubclass(enum_class, Enum):
                    annotations = extract_enum_annotations(enum_class)
                    
                    if annotations:
                        # Format the annotations as specified
                        enum_values = type_schema['enum']
                        annotation_text = f"Values:\n"

                        for value in enum_values:
                            annotation = annotations.get(value, value)
                            annotation_text += f"<{value}>: {annotation}\n"

                        # Store the annotation text for this enum type
                        enum_annotations_by_type[type_name] = annotation_text
    
                        # Add or append to the description
                        if 'description' in type_schema:
                            type_schema['description'] += "\n\n" + annotation_text
                        else:
                            type_schema['description'] = annotation_text
                            
                        type_schema['description'] = type_schema.get('description', '') + "\n\n" + 'structure: <|key|>: value'
    
    # Now add enum annotations to property fields that reference enums
    if enum_annotations_by_type and 'properties' in parameters:
        for prop_name, prop_schema in parameters['properties'].items():
            # Check if this property references an enum type
            if '$ref' in prop_schema:
                ref_path = prop_schema['$ref']
                # Extract the type name from the reference path (e.g., "#/$defs/StatusEnum" -> "StatusEnum")
                ref_type = ref_path.split('/')[-1]
                
                if ref_type in enum_annotations_by_type:
                    # Add the enum annotations to the property description
                    annotation_text = enum_annotations_by_type[ref_type]
                    if 'description' in prop_schema:
                        # Combine the field's description with the enum annotations in a more integrated way
                        field_desc = prop_schema['description']
                        # Check if the field description already ends with a period
                        if field_desc and not field_desc.endswith(('.', '!', '?', ':', ';')):
                            field_desc += "."
                        prop_schema['description'] = f"{field_desc}, Options: {annotation_text}"
                    else:
                        prop_schema['description'] = annotation_text

    parameters["required"] = sorted(
        k for k, v in parameters["properties"].items() if "default" not in v
    )

    if "description" not in schema:
        if docstring.short_description:
            schema["description"] = docstring.short_description
        else:
            schema["description"] = (
                f"Correctly extracted `{model.__name__}` with all "
                f"the required parameters with correct types"
            )

    return {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": parameters,
    }


@functools.lru_cache(maxsize=256)
def generate_anthropic_schema(model: type[BaseModel]) -> dict[str, Any]:
    """
    Generate Anthropic tool schema from a Pydantic model.

    Args:
        model: A Pydantic BaseModel subclass

    Returns:
        A dictionary in the format of Anthropic's tool schema
    """
    # Generate the Anthropic schema based on the OpenAI schema to avoid redundant schema generation
    openai_schema = generate_openai_schema(model)
    return {
        "name": openai_schema["name"],
        "description": openai_schema["description"],
        "input_schema": model.model_json_schema(),
    }


@functools.lru_cache(maxsize=256)
def generate_gemini_schema(model: type[BaseModel]) -> Any:
    """
    Generate Gemini function schema from a Pydantic model.

    Args:
        model: A Pydantic BaseModel subclass

    Returns:
        A Gemini FunctionDeclaration object

    Note:
        This function is deprecated. The google-generativeai library is being replaced by google-genai.
    """
    # This is kept for backward compatibility but deprecated
    warnings.warn(
        "generate_gemini_schema is deprecated. The google-generativeai library is being replaced by google-genai.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        import google.generativeai.types as genai_types

        # Use OpenAI schema
        openai_schema = generate_openai_schema(model)

        # Transform to Gemini format
        function = genai_types.FunctionDeclaration(
            name=openai_schema["name"],
            description=openai_schema["description"],
            parameters=map_to_gemini_function_schema(openai_schema["parameters"]),
        )

        return function
    except ImportError as e:
        raise ImportError(
            "google-generativeai is deprecated. Please install google-genai instead: pip install google-genai"
        ) from e
