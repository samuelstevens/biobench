CREATE TABLE IF NOT EXISTS reports (
    -- JSON blob
    model_config TEXT NOT NULL,

    task TEXT NOT NULL,
    posix INTEGER NOT NULL,

    -- Results
    mean_score FLOAT NOT NULL,
    confidence_lower FLOAT NOT NULL,
    confidence_upper FLOAT NOT NULL,

    -- Arbitrary JSON blobs
    args TEXT NOT NULL,
    report TEXT NOT NULL
);
