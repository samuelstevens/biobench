"""
Syncs all semprobe data between two machines.

Must run on the machine with remote access permission.
"""

import os
import pathlib
import shutil
import sqlite3
import subprocess

import beartype
import tyro


@beartype.beartype
def from_remote(ssh_host: str, remote_path: str, local_path: str):
    """
    Syncs all data from ssh_host:remote_path to local_path using rsync or scp, depending on what is available on your system.

    Args:
        ssh_host: The hostname or IP address of the remote machine to sync from. Can be a user@host, or a HostName found in your .ssh/config file.
        remote_path: The path on the remote machine to the results.sqlite database.
        local_path: The local destination directory where the database will be copied to.
    """

    # Create local directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)

    local_path = os.path.join(local_path, f"{ssh_host}-results.sqlite")

    if shutil.which("scp"):
        cmd = ["scp", f"{ssh_host}:{remote_path}", local_path]
    else:
        raise RuntimeError("scp not found in $PATH")

    # Execute the sync command
    subprocess.run(cmd, check=True)

    print(f"Synced to '{local_path}'.")


@beartype.beartype
def merge_local(
    db1: str,
    db2: str,
    out: str,
    schema_path: str = os.path.join("biobench", "schema.sql"),
    overwrite: bool = False,
):
    """
    Merge two SQLite databases using an existing schema file.

    Args:
        db1: Path to the first database
        db2: Path to the second database
        out: Path to the output merged database
        schema_path: Path to the schema SQL file
        overwrite: Whether to overwrite the output database if it exists
    """
    # Check if output exists
    output_path = pathlib.Path(out)
    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            raise FileExistsError(
                f"Output file {output_path} already exists. Use --overwrite to force."
            )

    # Connect to all databases
    db1 = sqlite3.connect(db1)
    db2 = sqlite3.connect(db2)
    output_db = sqlite3.connect(output_path)

    # Enable foreign keys
    output_db.execute("PRAGMA foreign_keys = ON")

    # Create tables in output database using the provided schema
    print(f"Creating tables in {output_path} using schema from {schema_path}...")
    with open(schema_path, "r") as schema_file:
        schema_sql = schema_file.read()
        output_db.executescript(schema_sql)

    result_id_mapping = {}

    # Define fixed columns for results table
    results_columns = [
        "task_name",
        "task_cluster",
        "task_subcluster",
        "n_train",
        "n_test",
        "sampling",
        "model_method",
        "model_org",
        "model_ckpt",
        "prompting",
        "cot_enabled",
        "parse_success_rate",
        "usd_per_answer",
        "classifier_type",
        "exp_cfg",
        "argv",
        "git_commit",
        "posix",
        "gpu_name",
        "hostname",
    ]
    columns_str = ", ".join(results_columns)

    # Copy data from results table
    print("Merging 'results' table...")

    # Helper function to insert results and track IDs
    @beartype.beartype
    def insert_results_from_db(source_db: sqlite3.Connection, db_name: str):
        query = f"SELECT id, {columns_str} FROM results"
        for row in source_db.execute(query).fetchall():
            old_id, *values = row

            placeholders = ", ".join(["?"] * len(values))
            output_cursor = output_db.cursor()
            output_cursor.execute(
                f"INSERT INTO results ({columns_str}) VALUES ({placeholders})", values
            )
            new_id = output_cursor.lastrowid

            # Store mapping from old id to new id
            result_id_mapping[(db_name, old_id)] = new_id

    # Insert from both databases
    insert_results_from_db(db1, "db1")
    insert_results_from_db(db2, "db2")
    output_db.commit()

    # Copy data from predictions table
    print("Merging 'predictions' table...")

    # Helper function to process and insert predictions
    def insert_predictions_from_db(source_db, db_name):
        cursor = source_db.cursor()
        cursor.execute(
            "SELECT img_id, score, n_train, info, result_id FROM predictions"
        )

        for img_id, score, n_train, info, old_result_id in cursor.fetchall():
            new_result_id = result_id_mapping.get((db_name, old_result_id))

            if new_result_id is None:
                print(
                    f"Warning: Could not find mapping for result_id {old_result_id} from {db_name}"
                )
                continue

            # Try to insert, but handle potential primary key conflicts
            try:
                output_db.execute(
                    "INSERT INTO predictions (img_id, score, n_train, info, result_id) VALUES (?, ?, ?, ?, ?)",
                    (img_id, score, n_train, info, new_result_id),
                )
            except sqlite3.IntegrityError as err:
                # If the error is a primary key constraint violation, we can skip
                if "UNIQUE constraint failed" in str(err):
                    print(
                        f"Skipping duplicate prediction entry: {img_id} with new result_id {new_result_id}"
                    )
                else:
                    raise

    # Insert from both databases
    insert_predictions_from_db(db1, "db1")
    insert_predictions_from_db(db2, "db2")

    # Commit changes and close connections
    output_db.commit()
    db1.close()
    db2.close()
    output_db.close()

    print(f"Merge completed successfully. Output saved to {output_path}")
    print(f"Merged {len(result_id_mapping)} results from the databases.")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "from-remote": from_remote,
        "merge-local": merge_local,
    })
