import click

from benchmarks.config import (
    CATALOG_REGISTRY,
    STANDARD_PAIRS,
    BenchmarkConfig,
)
from benchmarks.report import console_report, csv_report, json_report
from benchmarks.runner import run_repeated


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
@click.option(
    "--output",
    multiple=True,
    default=["console"],
    type=click.Choice(["console", "json", "csv"]),
    help="Output format(s).",
)
def run(catalog_a, catalog_b, catalog_a_path, catalog_b_path, radius, n_neighbors, repeat, n_workers, output):
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
    )
    results = run_repeated(config)
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
@click.option(
    "--output",
    multiple=True,
    default=["console", "json"],
    type=click.Choice(["console", "json", "csv"]),
    help="Output format(s).",
)
def sweep(catalog_a, catalog_b, radii, n_neighbors, n_workers, output):
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
        )
        results = run_repeated(config)
        all_results.extend(results)
        _output_results(results, output)

    click.echo(f"\nSweep complete: {len(all_results)} runs across {len(radius_list)} radii.")


@cli.command()
@click.option("--n-workers", default=None, type=int, help="Dask worker count.")
@click.option(
    "--output",
    multiple=True,
    default=["console", "json"],
    type=click.Choice(["console", "json", "csv"]),
    help="Output format(s).",
)
def suite(n_workers, output):
    """Run the standard benchmark suite (predefined catalog pairs)."""
    for a, b in STANDARD_PAIRS:
        click.echo(f"\n{'='*60}")
        click.echo(f"  {a} x {b}")
        click.echo(f"{'='*60}")
        config = BenchmarkConfig(
            catalog_a=a,
            catalog_b=b,
            n_workers=n_workers,
        )
        results = run_repeated(config)
        _output_results(results, output)

    click.echo(f"\nSuite complete: {len(STANDARD_PAIRS)} pairs benchmarked.")


@cli.command("list")
def list_catalogs():
    """List all catalogs in the registry."""
    for name, url in sorted(CATALOG_REGISTRY.items()):
        click.echo(f"  {name:20s} {url}")


if __name__ == "__main__":
    cli()
