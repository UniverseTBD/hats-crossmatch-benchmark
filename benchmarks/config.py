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
    test: bool = False  # Cone-search to small sky region for fast iteration
    test_radius_deg: float = 1.0  # Cone radius when test=True


# SDSS field center — dense region with good multi-survey coverage
TEST_CONE_RA = 150.0   # degrees (in COSMOS/SDSS Stripe 82 area)
TEST_CONE_DEC = 2.0    # degrees


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
    # Small-large: SNe surveys against wide-field
    ("cfa_cfa3", "sdss"),
    ("swift_sne_ia", "gaia"),
    # Medium-large: Galaxy Zoo uses SDSS imaging
    ("gz10", "sdss"),
    # Large-large: wide-field surveys
    ("sdss", "gaia"),
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
