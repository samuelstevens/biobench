PRAGMA journal_mode = WAL;   -- Concurrent reads/writes
PRAGMA synchronous = NORMAL; -- Good balance speed/safety
PRAGMA foreign_keys = ON;    -- Enforce FK constraints
PRAGMA busy_timeout = 5000;  -- Wait up to 5000ms before throwing timeout errors
PRAGMA strict = ON;          -- Enforce strict type checking (SQLite ≥ 3.37)
PRAGMA encoding = 'UTF-8';   -- Consistent text encoding


CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Task information
    task_name TEXT NOT NULL,

    -- Model information
    model_cfg TEXT NOT NULL, -- JSON blob

    exp_cfg TEXT NOT NULL,  -- JSON blob with full experiment configuration

    -- Metadata fields
    argv TEXT NOT NULL,  -- Command used to get this report (JSON array)
    git_commit TEXT NOT NULL,  -- Git commit hash
    posix INTEGER NOT NULL,  -- POSIX timestamp
    gpu_name TEXT,  -- Name of the GPU that ran this experiment
    hostname TEXT NOT NULL  -- Machine hostname that ran this experiment
);

CREATE TABLE IF NOT EXISTS predictions (
    img_id TEXT NOT NULL,  -- ID used to find the original image/example
    score REAL NOT NULL,  -- Test score; typically 0 or 1 for classification tasks
    n_train INTEGER NOT NULL,  -- Number of training examples used in this prediction
    info TEXT NOT NULL,  -- JSON blob with additional information (original class, true label, etc.)

    -- Foreign key to link to the results table
    result_id INTEGER NOT NULL,

    PRIMARY KEY (img_id, result_id),
    FOREIGN KEY (result_id) REFERENCES results(id)  ON DELETE CASCADE
);
