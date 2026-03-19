"""Streaming crossmatch client that returns an HF IterableDataset.

Wraps lsdb's CatalogStream (which yields pandas DataFrames) into a
datasets.IterableDataset (which yields dicts row-by-row), giving
downstream code a unified interface whether data comes from a live
lsdb crossmatch or a pre-computed HuggingFace dataset.
"""

from __future__ import annotations

from typing import Any

import lsdb
from datasets import Features, IterableDataset, Value
from lsdb.streams import CatalogStream


def _iter_rows(xmatch):
    """Yield one dict per row from the crossmatch CatalogStream."""
    stream = CatalogStream(catalog=xmatch, shuffle=False)
    for chunk in stream:
        for record in chunk.to_dict("records"):
            yield record


def stream_crossmatch(
    url_a: str,
    url_b: str,
    *,
    radius_arcsec: float = 1.0,
    n_neighbors: int = 1,
    suffixes: tuple[str, str] = ("_a", "_b"),
    search_filter: Any | None = None,
    storage_options_a: dict | None = None,
    storage_options_b: dict | None = None,
) -> IterableDataset:
    """Stream crossmatch results as an HF IterableDataset.

    Parameters
    ----------
    url_a : str
        HATS catalog URL (e.g. "hf://datasets/UniverseTBD/mmu_sdss_sdss").
    url_b : str
        HATS catalog URL.
    radius_arcsec : float
        Crossmatch radius in arcseconds.
    n_neighbors : int
        Maximum number of neighbors to return per source.
    suffixes : tuple[str, str]
        Column suffixes to disambiguate the two catalogs.
    search_filter : optional
        lsdb search filter (e.g. ``lsdb.ConeSearch(...)``).
    storage_options_a : dict or None
        Storage options for catalog A (e.g. ``{"anon": True}`` for S3).
    storage_options_b : dict or None
        Storage options for catalog B.

    Returns
    -------
    datasets.IterableDataset
        An iterable dataset where each element is a dict representing
        one crossmatch row.
    """
    kwargs_a: dict[str, Any] = {}
    kwargs_b: dict[str, Any] = {}

    if search_filter is not None:
        kwargs_a["search_filter"] = search_filter
        kwargs_b["search_filter"] = search_filter

    if storage_options_a is not None:
        kwargs_a["storage_options"] = storage_options_a
    elif url_a.startswith("s3://"):
        kwargs_a["storage_options"] = {"anon": True}

    if storage_options_b is not None:
        kwargs_b["storage_options"] = storage_options_b
    elif url_b.startswith("s3://"):
        kwargs_b["storage_options"] = {"anon": True}

    cat_a = lsdb.open_catalog(url_a, **kwargs_a)
    cat_b = lsdb.open_catalog(url_b, **kwargs_b)

    xmatch = cat_a.crossmatch(
        cat_b,
        n_neighbors=n_neighbors,
        radius_arcsec=radius_arcsec,
        suffixes=suffixes,
        suffix_method="all_columns",
    )

    return IterableDataset.from_generator(_iter_rows, gen_kwargs={"xmatch": xmatch})
