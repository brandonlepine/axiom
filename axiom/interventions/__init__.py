"""Intervention runners (residual/head patching, ablation, knockout, steering, MLP).

This package defines the :class:`~axiom.interventions.base.InterventionRunner` contract
and the concrete runners ported from the validated upstream scripts. Each is a
self-contained class instantiable and testable in isolation (CLAUDE.md, "Use
object-oriented pipeline components").
"""
from axiom.interventions.base import InterventionRunner, StepResult
from axiom.interventions.residual import ResidualPatcher

__all__ = ["InterventionRunner", "ResidualPatcher", "StepResult"]
