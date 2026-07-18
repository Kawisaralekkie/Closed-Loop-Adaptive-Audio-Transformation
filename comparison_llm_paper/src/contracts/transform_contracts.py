"""Transform data contracts for blurring recipes and results.

Defines Pydantic models for recipe references, transform parameters,
and transform results used by MidBandAttenuationTool and StrongBlurringTool.

Requirements: 16.1
"""

from __future__ import annotations

from pydantic import BaseModel


class TransformRecipeRef(BaseModel):
    """Reference to a specific blurring recipe and version."""

    recipe_name: str  # "RECIPE_MID_BAND_ATTEN" or "RECIPE_LOWPASS_HIGHPASS_MIX"
    version: str


class TransformParams(BaseModel):
    """Parameters for a single blurring trial."""

    recipe_ref: TransformRecipeRef
    params: dict  # recipe-specific parameters
    trial: int = 0


class TransformResult(BaseModel):
    """Output of a blurring tool execution."""

    chunk_id: str
    recipe_ref: TransformRecipeRef
    params: TransformParams
    blurred_wav_path: str
    trial: int
    success: bool
