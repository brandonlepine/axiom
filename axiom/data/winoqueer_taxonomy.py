"""WinoQueer identity taxonomy: annotate pairs with axis / identity / is_umbrella.

The WinoQueer candidate pool carries the raw ``Gender_ID_x`` / ``Gender_ID_y`` identity
strings but not the segmentation columns the cohort schema and analyses need. This is the
single source of truth that maps an identity to its construct axis (ported from the prior
repo's ``winoqueer_identity_taxonomy``). The axis is grounded in the data: the control
word (``Gender_ID_y``) confirms it.

  sexual_orientation : Asexual, Bisexual, Gay, Lesbian, Pansexual  (controls Straight/Heterosexual)
  gender_identity    : Transgender, NB                             (controls Cis/Cisgender)
  umbrella           : Queer, LGBTQ                                (pair with both controls)
"""
from __future__ import annotations

import pandas as pd

IDENTITY_AXIS: dict[str, str] = {
    "Asexual": "sexual_orientation",
    "Bisexual": "sexual_orientation",
    "Gay": "sexual_orientation",
    "Lesbian": "sexual_orientation",
    "Pansexual": "sexual_orientation",
    "Transgender": "gender_identity",
    "NB": "gender_identity",
    "Queer": "umbrella",
    "LGBTQ": "umbrella",
}
CONTROL_AXIS: dict[str, str] = {
    "Straight": "sexual_orientation",
    "Heterosexual": "sexual_orientation",
    "Cis": "gender_identity",
    "Cisgender": "gender_identity",
}
UMBRELLA: set[str] = {"Queer", "LGBTQ"}


def axis_mismatch_count(df: pd.DataFrame) -> int:
    """# specific (non-umbrella) pairs whose control word's axis disagrees with the identity's."""
    ax = df["Gender_ID_x"].map(IDENTITY_AXIS)
    cy = df["Gender_ID_y"].map(CONTROL_AXIS)
    mask = ax.isin(["sexual_orientation", "gender_identity"]) & cy.notna() & (ax != cy)
    return int(mask.sum())


def annotate(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Return a copy of ``df`` with ``identity``, ``axis``, ``is_umbrella`` added.

    ``identity = Gender_ID_x``; ``axis`` via :data:`IDENTITY_AXIS`; ``is_umbrella`` for
    Queer/LGBTQ. With ``strict=True`` (default), raises on any ``Gender_ID_x`` outside the
    taxonomy or any specific identity paired with a control of the wrong axis -- a schema
    mismatch is a catastrophic failure, not a silent one (CLAUDE.md).
    """
    out = df.copy()
    out["identity"] = out["Gender_ID_x"].astype(str)
    out["axis"] = out["identity"].map(IDENTITY_AXIS)
    out["is_umbrella"] = out["identity"].isin(UMBRELLA)
    if out["axis"].isna().any():
        unknown = sorted(out.loc[out["axis"].isna(), "identity"].unique())
        msg = f"Gender_ID_x values not in the WinoQueer taxonomy: {unknown}"
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)
    if strict:
        m = axis_mismatch_count(df)
        if m:
            raise ValueError(f"{m} specific-identity pairs have a control whose axis disagrees.")
    return out
