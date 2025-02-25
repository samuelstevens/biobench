CREATE TABLE IF NOT EXISTS results (
    -- Experiment metadata
    posix INTEGER NOT NULL,  -- POSIX timestamp

    -- Task information
    task_name TEXT NOT NULL,  -- "newt", "iwildcam", etc.
    n_train INTEGER NOT NULL,  -- Number of train samples *actually* used (1, 3, 10, 30, etc.)
    n_test INTEGER NOT NULL, -- Number of test samples actually used
    sampling TEXT NOT NULL,  -- "uniform" or "class_balanced"

    -- Model information
    model_method TEXT NOT NULL,  -- "mllm" or "cvml"
    model_org TEXT NOT NULL,
    model_ckpt TEXT NOT NULL,

    -- Common metrics
    mean_score REAL NOT NULL,  -- Primary metric
    confidence_lower REAL NOT NULL,
    confidence_upper REAL NOT NULL,

    -- MLLM-specific fields
    prompting TEXT,  -- "single" or "multi"
    cot_enabled INTEGER,  -- Boolean as 0 or 1
    parse_success_rate REAL, -- Parse success rate as a value in [0, 1]
    usd_per_answer REAL, -- Cost in USD per answer

    -- CVML-specific fields
    classifier_type TEXT,  -- "knn", "svm", or "ridge"

    -- Flexible storage for complete configurations and detailed results
    exp_cfg TEXT NOT NULL,  -- JSON blob with full experiment configuration
    report TEXT NOT NULL  -- JSON blob with detailed results
);
