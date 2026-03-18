from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    catalog_a: str
    catalog_b: str
    radius_arcsec: float = 1.0
    n_neighbors: int = 1
    suffixes: tuple[str, str] = ("_a", "_b")
    repeat: int = 1
    n_workers: int | None = None


CATALOG_REGISTRY = {
    # Large catalogs (>100k rows)
    "plasticc": "hf://datasets/UniverseTBD/mmu_plasticc",
    "desi": "hf://datasets/UniverseTBD/mmu_desi_edr_sv3",
    "tess": "hf://datasets/UniverseTBD/mmu_tess_spoc",
    "btsbot": "hf://datasets/UniverseTBD/mmu_btsbot",
    "sdss": "hf://datasets/UniverseTBD/mmu_sdss_sdss",
    "hsc": "hf://datasets/UniverseTBD/mmu_hsc_pdr3_dud_22.5",
    "desi_provabgs": "hf://datasets/UniverseTBD/mmu_desi_provabgs",
    "chandra": "hf://datasets/UniverseTBD/mmu_chandra_spectra",
    "gaia": "hf://datasets/UniverseTBD/mmu_gaia_gaia",
    # Medium catalogs (1k-100k rows)
    "jwst_primer": "hf://datasets/UniverseTBD/mmu_jwst_primer_cosmos",
    "vipers_w1": "hf://datasets/LSDB/mmu_vipers_w1",
    "jwst_gdn": "hf://datasets/UniverseTBD/mmu_jwst_gdn",
    "vipers_w4": "hf://datasets/LSDB/mmu_vipers_w4",
    "jwst_ceers": "hf://datasets/UniverseTBD/mmu_jwst_ceers",
    "gz10": "hf://datasets/UniverseTBD/mmu_gz10",
    "yse": "hf://datasets/UniverseTBD/mmu_yse_dr1",
    # Small catalogs (<1k rows)
    "ps1_sne_ia": "hf://datasets/UniverseTBD/mmu_ps1_sne_ia",
    "des_y3_sne_ia": "hf://datasets/UniverseTBD/mmu_des_y3_sne_ia",
    "foundation": "hf://datasets/UniverseTBD/mmu_foundation",
    "snls": "hf://datasets/UniverseTBD/mmu_snls",
    "swift_sne_ia": "hf://datasets/UniverseTBD/mmu_swift_sne_ia",
    "cfa_cfa3": "hf://datasets/UniverseTBD/mmu_cfa_cfa3",
    "csp": "hf://datasets/UniverseTBD/mmu_csp_csp",
    "cfa_cfa4": "hf://datasets/UniverseTBD/mmu_cfa_cfa4",
    "cfa_snii": "hf://datasets/UniverseTBD/mmu_cfa_snii",
    "cfa_seccsn": "hf://datasets/UniverseTBD/mmu_cfa_seccsn",
}

STANDARD_PAIRS = [
    # Small-small: SNe surveys with sky overlap
    ("foundation", "des_y3_sne_ia"),
    ("snls", "ps1_sne_ia"),
    # Medium-medium
    ("vipers_w1", "gz10"),
    # Medium-large
    ("jwst_ceers", "sdss"),
]


def resolve_catalog(name_or_url: str) -> str:
    """Resolve a catalog short name to its HF URL, or return the URL as-is."""
    if name_or_url.startswith("hf://"):
        return name_or_url
    if name_or_url not in CATALOG_REGISTRY:
        raise ValueError(
            f"Unknown catalog '{name_or_url}'. "
            f"Available: {', '.join(sorted(CATALOG_REGISTRY))}"
        )
    return CATALOG_REGISTRY[name_or_url]
