CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Task information
    task_name TEXT NOT NULL,  -- "newt", "iwildcam", etc.
    task_cluster TEXT,
    task_subcluster TEXT,
    n_train INTEGER NOT NULL,  -- Number of train samples *actually* used (1, 3, 10, 30, etc.)
    n_test INTEGER NOT NULL, -- Number of test samples actually used
    sampling TEXT NOT NULL,  -- "uniform" or "class_balanced"

    -- Model information
    model_method TEXT NOT NULL,  -- "mllm" or "cvml"
    model_org TEXT NOT NULL,
    model_ckpt TEXT NOT NULL,

    -- MLLM-specific fields
    prompting TEXT,  -- "single" or "multi"
    cot_enabled INTEGER,  -- Boolean as 0 or 1
    parse_success_rate REAL, -- Parse success rate as a value in [0, 1]
    usd_per_answer REAL, -- Cost in USD per answer

    -- CVML-specific fields
    classifier_type TEXT,  -- "knn", "svm", or "ridge"

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
    FOREIGN KEY (result_id) REFERENCES results(id)
);
