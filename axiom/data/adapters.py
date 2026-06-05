"""The DatasetAdapter interface: how a dataset enters the pipeline uniformly.

Adding a new dataset (a new Wino-style variant, a new benchmark transformed to the
continuation-scoring format) should mean implementing one adapter -- not editing every
runner (CLAUDE.md, "Interfaces for extensibility"). An adapter knows how to load its
frozen cohort and which schema it satisfies; everything downstream consumes the
validated DataFrame and the schema's identity-column names.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from axiom.data.schemas import CohortSchema, validate_cohort


class DatasetAdapter(ABC):
    """Base class for a dataset's entry point into the pipeline.

    Concrete adapters set :attr:`schema` and implement :meth:`cohort_path`. The default
    :meth:`load_cohort` reads the CSV in file order (cohorts are pre-sorted and frozen)
    and validates it against the schema before any analysis touches it.
    """

    #: The cohort schema this dataset satisfies (set by the subclass).
    schema: CohortSchema

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id

    @abstractmethod
    def cohort_path(self) -> Path:
        """Absolute path to this dataset's frozen ``cohort.csv``."""
        raise NotImplementedError

    def load_cohort(self, validate: bool = True) -> pd.DataFrame:
        """Load the frozen cohort in file order; validate against the schema by default."""
        df = pd.read_csv(self.cohort_path())
        if validate:
            validate_cohort(df, self.schema)
        return df

    @property
    def identity_columns(self) -> tuple[str, str]:
        """The (target, reference) identity column names for this dataset's schema."""
        return self.schema.identity_x_col, self.schema.identity_y_col
