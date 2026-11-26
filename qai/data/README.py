"""
qai.data — Unified Data Manipulation Library.

Everything is accessible directly from qai.data — no .pro or .multimodal
sub-modules needed by the user. All classes are re-exported here.

  TABULAR:
    DataDataset          — core dataset container (load, manipulate, export)
    DataCleaner          — automated cleaning (missing, outliers, duplicates)
    FeatureEngineer      — auto-generate features (interactions, polynomials)
    DataAugmentor        — SMOTE, jitter, mixup, cutout
    DataTransformer      — encode, scale, normalize
    DataSplitter         — train/val/test + stratified + time-series
    DataInspector        — deep profiling
    DataPipeline2        — chain operations
    DataExporter         — export to 11+ formats
    DataVisualizer       — auto-generate charts
    DataVisualizer2      — advanced viz for all data types

  MULTIMODAL:
    TextProcessor        — clean, tokenize, sentiment, keywords, vocab
    ImageProcessor       — load, resize, augment, normalize, edge detection
    VideoProcessor       — frame extraction, scene detection, stats
    AudioProcessor       — spectrogram, MFCC, mel-spectrogram, stats
    GraphProcessor       — adjacency, node features, k-hop subgraph
    GeospatialProcessor  — Haversine, coordinates, bounding box, clustering
    MultimodalPipeline   — auto-detect type + process any file

  ENTERPRISE:
    DatasetProfiler      — rich per-column profiling
    StreamingDataset     — chunked iterator with backpressure
    DatasetJoiner        — SQL-style joins
    TimeSeriesToolkit    — resample, seasonality, STL decomposition
    SchemaEnforcer       — type validation + coercion
    DataPoisoningDetector— statistical + signature detection
    DriftDetector        — KS / PSI / chi-square drift
    SyntheticDataFactory — tabular, time-series, text, image generation
    FeatureStore         — offline + online feature lookups
    DataLineage          — track provenance
    DataCatalog          — searchable registry
    SmartSampler         — stratified, imbalance-aware
    DataValidator        — great-expectations-style rules
    SmartCache           — disk-backed memoization
    AnnotationLab        — weak supervision + label noise detection
    TimeSeriesPrep       — lag features, rolling stats, datetime features
    TextPreprocessor     — clean, tokenize, stopwords, vocab

  PIPELINE:
    DataPipeline         — original pipeline (from earlier version)
    DatabaseConnector    — database connection
    DocumentReader       — document parsing
    SchemaBuilder        — build schemas
    DataVersioner        — version tracking

Usage:
    import qai
    df = qai.data.load("data.csv")
    cleaned, report = qai.data.DataCleaner.clean(df)
    enhanced, new_feats = qai.data.FeatureEngineer.auto_features(cleaned)
    qai.data.DataVisualizer2.visualize(enhanced, save_path="report.png")
    result = qai.data.MultimodalPipeline.process("image.jpg")
"""
