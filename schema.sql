CREATE TABLE IF NOT EXISTS reports (
    -- 
    model_org TEXT NOT NULL,
    model_ckpt TEXT NOT NULL,
    task_name TEXT NOT NULL,
    posix INTEGER NOT NULL,
    -- Results
    mean_score FLOAT NOT NULL,
    confidence_lower FLOAT NOT NULL,
    confidence_upper FLOAT NOT NULL,

    -- Arbitrary JSON blobs
    launcher TEXT NOT NULL,
    report TEXT NOT NULL
);
