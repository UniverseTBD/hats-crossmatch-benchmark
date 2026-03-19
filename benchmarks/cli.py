import warnings

import click

warnings.filterwarnings("ignore", message="The behavior of array concatenation with empty entries")

from benchmarks.config import (
    CATALOG_REGISTRY,
    S3_PAIRS,
    STANDARD_PAIRS,
    STREAM_HF_REPOS,
    BenchmarkConfig,
)
from benchmarks.report import console_report, csv_report, json_report
from benchmarks.runner import run_repeated
from benchmarks.stream_runner import run_stream_repeated


def _output_results(results, output_formats):
    for fmt in output_formats:
        if fmt == "console":
            console_report(results)
        elif fmt == "json":
            json_report(results)
        elif fmt == "csv":
            csv_report(results)


@click.group()
def cli():
    """HATS crossmatch benchmark tool."""
    pass


@cli.command()
@click.option(
    "--catalog-a",
    required=False,
    help="Catalog A short name from registry.",
)
@click.option(
    "--catalog-b",
    required=False,
    help="Catalog B short name from registry.",
)
@click.option(
    "--catalog-a-path",
    required=False,
    help="Direct HF URL for catalog A (overrides --catalog-a).",
)
@click.option(
    "--catalog-b-path",
    required=False,
    help="Direct HF URL for catalog B (overrides --catalog-b).",
)
@click.option("--radius", default=1.0, help="Crossmatch radius in arcsec.")
@click.option("--n-neighbors", default=1, help="Number of neighbors to find.")
@click.option("--repeat", default=1, help="Number of times to repeat the benchmark.")
@click.option("--n-workers", default=None, type=int, help="Dask worker count.")
@click.option("--test", is_flag=True, help="Test mode: cone-search a small sky region, synchronous scheduler.")
@click.option(
    "--output",
    multiple=True,
    default=["console"],
    type=click.Choice(["console", "json", "csv"]),
    help="Output format(s).",
)
def run(catalog_a, catalog_b, catalog_a_path, catalog_b_path, radius, n_neighbors, repeat, n_workers, test, output):
    """Run a single crossmatch benchmark."""
    a = catalog_a_path or catalog_a
    b = catalog_b_path or catalog_b
    if not a or not b:
        raise click.UsageError("Must provide both catalogs via --catalog-a/--catalog-b or --catalog-a-path/--catalog-b-path")

    config = BenchmarkConfig(
        catalog_a=a,
        catalog_b=b,
        radius_arcsec=radius,
        n_neighbors=n_neighbors,
        repeat=repeat,
        n_workers=n_workers,
        test=test,
    )
    results = run_repeated(config)
    _output_results(results, output)


@cli.command()
@click.option("--catalog-a", required=False, help="Catalog A short name from registry.")
@click.option("--catalog-b", required=False, help="Catalog B short name from registry.")
@click.option("--catalog-a-path", required=False, help="Direct HF URL for catalog A (overrides --catalog-a).")
@click.option("--catalog-b-path", required=False, help="Direct HF URL for catalog B (overrides --catalog-b).")
@click.option("--radius", default=1.0, help="Crossmatch radius in arcsec.")
@click.option("--n-neighbors", default=1, help="Number of neighbors to find.")
@click.option("--repeat", default=1, help="Number of times to repeat the benchmark.")
@click.option("--n-workers", default=None, type=int, help="Dask worker count.")
@click.option("--test", is_flag=True, help="Test mode: cone-search a small sky region.")
@click.option(
    "--output",
    multiple=True,
    default=["console"],
    type=click.Choice(["console", "json", "csv"]),
    help="Output format(s).",
)
def stream(catalog_a, catalog_b, catalog_a_path, catalog_b_path, radius, n_neighbors, repeat, n_workers, test, output):
    """Run a streaming crossmatch benchmark (partition-by-partition)."""
    a = catalog_a_path or catalog_a
    b = catalog_b_path or catalog_b
    if not a or not b:
        raise click.UsageError("Must provide both catalogs via --catalog-a/--catalog-b or --catalog-a-path/--catalog-b-path")

    config = BenchmarkConfig(
        catalog_a=a,
        catalog_b=b,
        radius_arcsec=radius,
        n_neighbors=n_neighbors,
        repeat=repeat,
        n_workers=n_workers,
        test=test,
        mode="stream",
    )
    results = run_stream_repeated(config)
    _output_results(results, output)


@cli.command("stream-hf")
@click.option("--repo-id", required=True, help="HuggingFace dataset repo ID (e.g. UniverseTBD/sdss_x_plasticc).")
@click.option("--repeat", default=1, help="Number of times to repeat the benchmark.")
@click.option(
    "--output",
    multiple=True,
    default=["console"],
    type=click.Choice(["console", "json", "csv"]),
    help="Output format(s).",
)
def stream_hf(repo_id, repeat, output):
    """Run a streaming benchmark reading pre-computed results from HF Datasets."""
    config = BenchmarkConfig(
        catalog_a=repo_id,
        catalog_b=repo_id,
        repeat=repeat,
        mode="stream-hf",
        hf_repo_id=repo_id,
    )
    results = run_stream_repeated(config)
    _output_results(results, output)


@cli.command()
@click.option("--catalog-a", required=True, help="Catalog A short name or HF URL.")
@click.option("--catalog-b", required=True, help="Catalog B short name or HF URL.")
@click.option(
    "--radii",
    required=True,
    help="Comma-separated list of radii in arcsec (e.g. 0.1,0.5,1.0,2.0,5.0).",
)
@click.option("--n-neighbors", default=1, help="Number of neighbors.")
@click.option("--n-workers", default=None, type=int, help="Dask worker count.")
@click.option("--test", is_flag=True, help="Test mode: cone-search a small sky region, synchronous scheduler.")
@click.option(
    "--output",
    multiple=True,
    default=["console", "json"],
    type=click.Choice(["console", "json", "csv"]),
    help="Output format(s).",
)
def sweep(catalog_a, catalog_b, radii, n_neighbors, n_workers, test, output):
    """Run a parameter sweep varying crossmatch radius."""
    radius_list = [float(r.strip()) for r in radii.split(",")]
    all_results = []
    for radius in radius_list:
        click.echo(f"\n{'='*60}")
        click.echo(f"  Radius: {radius} arcsec")
        click.echo(f"{'='*60}")
        config = BenchmarkConfig(
            catalog_a=catalog_a,
            catalog_b=catalog_b,
            radius_arcsec=radius,
            n_neighbors=n_neighbors,
            n_workers=n_workers,
            test=test,
        )
        results = run_repeated(config)
        all_results.extend(results)
        _output_results(results, output)

    click.echo(f"\nSweep complete: {len(all_results)} runs across {len(radius_list)} radii.")


@cli.command()
@click.option("--n-workers", default=None, type=int, help="Dask worker count.")
@click.option("--test", is_flag=True, help="Test mode: cone-search a small sky region, synchronous scheduler.")
@click.option(
    "--mode",
    default="compute",
    type=click.Choice(["compute", "stream"]),
    help="Benchmark mode: compute (batch) or stream.",
)
@click.option(
    "--output",
    multiple=True,
    default=["console", "json"],
    type=click.Choice(["console", "json", "csv"]),
    help="Output format(s).",
)
def suite(n_workers, test, mode, output):
    """Run the standard benchmark suite (predefined catalog pairs)."""
    for a, b in STANDARD_PAIRS:
        click.echo(f"\n{'='*60}")
        click.echo(f"  {a} x {b}")
        click.echo(f"{'='*60}")
        config = BenchmarkConfig(
            catalog_a=a,
            catalog_b=b,
            n_workers=n_workers,
            test=test,
            mode=mode,
        )
        if mode == "stream":
            results = run_stream_repeated(config)
        else:
            results = run_repeated(config)
        _output_results(results, output)

    click.echo(f"\nSuite complete: {len(STANDARD_PAIRS)} pairs benchmarked.")


@cli.command("s3-suite")
@click.option("--n-workers", default=None, type=int, help="Dask worker count.")
@click.option("--test", is_flag=True, help="Test mode: cone-search a small sky region, synchronous scheduler.")
@click.option(
    "--mode",
    default="compute",
    type=click.Choice(["compute", "stream"]),
    help="Benchmark mode: compute (batch) or stream.",
)
@click.option(
    "--output",
    multiple=True,
    default=["console", "json"],
    type=click.Choice(["console", "json", "csv"]),
    help="Output format(s).",
)
def s3_suite(n_workers, test, mode, output):
    """Run the S3 benchmark suite (ZTF DR23, PS1 DR2, Gaia DR3)."""
    for a, b in S3_PAIRS:
        click.echo(f"\n{'='*60}")
        click.echo(f"  {a} x {b}")
        click.echo(f"{'='*60}")
        config = BenchmarkConfig(
            catalog_a=a,
            catalog_b=b,
            n_workers=n_workers,
            test=test,
            mode=mode,
        )
        if mode == "stream":
            results = run_stream_repeated(config)
        else:
            results = run_repeated(config)
        _output_results(results, output)

    click.echo(f"\nS3 suite complete: {len(S3_PAIRS)} pairs benchmarked.")


@cli.command("stream-hf-suite")
@click.option("--repeat", default=1, help="Number of times to repeat each benchmark.")
@click.option(
    "--output",
    multiple=True,
    default=["console", "json"],
    type=click.Choice(["console", "json", "csv"]),
    help="Output format(s).",
)
def stream_hf_suite(repeat, output):
    """Run the stream-hf benchmark suite (predefined HF repos)."""
    for repo_id in STREAM_HF_REPOS:
        click.echo(f"\n{'='*60}")
        click.echo(f"  {repo_id}")
        click.echo(f"{'='*60}")
        config = BenchmarkConfig(
            catalog_a=repo_id,
            catalog_b=repo_id,
            repeat=repeat,
            mode="stream-hf",
            hf_repo_id=repo_id,
        )
        results = run_stream_repeated(config)
        _output_results(results, output)

    click.echo(f"\nStream-HF suite complete: {len(STREAM_HF_REPOS)} repos benchmarked.")


@cli.command("list")
def list_catalogs():
    """List all catalogs in the registry."""
    for name, url in sorted(CATALOG_REGISTRY.items()):
        click.echo(f"  {name:20s} {url}")


if __name__ == "__main__":
    cli()
