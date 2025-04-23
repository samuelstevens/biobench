import datetime
import json
import pathlib


def main(
    coverage_path: str = "coverage.json",
    test_report_path: str = "pytest.json",
    out_fpath: str = "REGRESSIONS.md",
):
    """
    Args:
        coverage_path: Pytest-cov JSON coverage report.
        test_report_path: Pytest JSON test report.
        out_fpath: Output summary markdown file.
    """
    # Load failing tests
    with open(test_report_path, "r") as f:
        report = json.load(f)

    failures = [
        t["nodeid"] for t in report.get("tests", []) if t["outcome"] == "failed"
    ]

    # Load coverage data
    with open(coverage_path, "r") as f:
        coverage = json.load(f)

    covered = coverage["totals"]["covered_lines"]
    total = coverage["totals"]["num_statements"]
    percent = coverage["totals"]["percent_covered"]

    # Compose report
    out = [f"# Regressions\n\nLast checked: {datetime.date.today().isoformat()}\n"]

    if failures:
        out.append(f"# {len(failures)} failing test(s)\n")
        for name in sorted(failures):
            out.append(f"- {name}")
    else:
        out.append("**All tests passed**")

    out.append(f"# Coverage\n\nCoverage: {covered}/{total} lines ({percent:.1f}%)\n")

    pathlib.Path(out_fpath).write_text("\n".join(out))


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
