"""Intervention runners (residual/head patching, ablation, knockout, steering, MLP).

This package defines the :class:`~axiom.interventions.base.InterventionRunner` contract.
Concrete runners are ported one at a time from the validated upstream scripts; each is a
self-contained class instantiable and testable in isolation (CLAUDE.md, "Use
object-oriented pipeline components"). Until a runner is ported it does not exist here --
there are no behavior-claiming stubs.
"""
from axiom.interventions.base import InterventionRunner, StepResult

__all__ = ["InterventionRunner", "StepResult"]
