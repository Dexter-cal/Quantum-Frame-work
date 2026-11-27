"""
qai -- unified machine learning and artificial intelligence framework.

    import qai
    model = qai.build(type="regression")
    model.train(X, y)
    model.predict(x)

Provides unified single-import access to modeling, pipelines, datasets,
cross-validation, autotuning, benchmark, persistence, quantization, evaluation metrics,
explainability, synthetic generators, hardware profiling, system utilities, math & activation functions,
serving, preprocessing, governance, tracking, distributed multi-node, ONNX export, NAS, federated learning,
quantum circuit simulation, qai.data, qai.runtime, vision, audio, nlp, graph, time_series, interpretability, and multi-target stacking modules.
"""

from __future__ import annotations

# Formal version
__version__ = "1.8.7"
__version_info__ = (1, 8, 7)

from .core import build, Model, logic, LogicError, CompatibilityError, InputValidationError
from .core.logic import save_logic_source, load_logic_source
from .core.environment import GridWorld
from .core.dataset import Dataset, StreamingEventConnector
from .core.serve import serve, build_flask_app
from .core.pipeline import Pipeline
from .core.cross_validation import cross_validate, k_fold_split
from .core.ensemble import VotingEnsemble
from .core.autotune import AutoTuner, autotune, bayesian_optimize
from .core.benchmark import benchmark
from .core.persistence import save_model, load_model
from .core.quantize import compress_model, prune_structured_sparsity
from .core.distributed import init_distributed_context, DistributedDataParallelWrapper
from .core.onnx import export_to_onnx
from .core.nas import search_architecture
from .core.federated import FederatedServer, federated_averaging
from .core.metrics import (
    classification_report, confusion_matrix, mean_absolute_error, r2_score, log_loss, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score, silhouette_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, mean_squared_log_error, mean_absolute_percentage_error,
    median_absolute_error, explained_variance_score, max_error, brier_score, expected_calibration_error,
    precision_recall_curve, roc_curve, det_curve, coverage_error, label_ranking_average_precision_score,
    zero_one_loss, hamming_loss, jaccard_score, multilabel_confusion_matrix, brier_score_loss
)
from .core.explainability import feature_importance, permutation_importance, partial_dependence, shapley_value_approximation
from .core.generators import make_classification, make_regression, make_blobs
from .core.hardware import get_hardware_info, get_optimal_device
from .techniques import register_technique, Technique
from .learning import (
    get_learning_technique, LearningTechnique, MultiOutputClassifier, MultiOutputRegressor,
    ClassifierChain, StackingClassifier, StackingRegressor
)

# Advanced sub-packages & utilities
from .preprocessing import (
    StandardScaler, LabelEncoder, SimpleImputer, PolynomialFeatures, MinMaxScaler, RobustScaler,
    Normalizer, OneHotEncoder, Binarizer, VarianceThreshold, SelectKBest, MaxAbsScaler, OrdinalEncoder,
    TargetEncoder, KBinsDiscretizer, FunctionTransformer, ColumnTransformer, QuantileTransformer,
    PowerTransformer, RecursiveFeatureElimination, SelectFromModel, clean_dataset
)
from .governance import (
    detect_drift, weight_distance, calibration_curve, demographic_parity_difference,
    equalized_odds_difference, disparate_impact_ratio, fairness_audit, watermark_model, verify_watermark
)
from .tracking import ExperimentTracker, log_telemetry, sync_tensorboard
from .mechanics import QuantumCircuitSimulator, quantum_expectation, ReduceLROnPlateau

# Data, System, Math, Runtime, Vision, Audio, NLP, Graph, Time-Series sub-packages
from . import data
from . import system
from . import math
from . import runtime
from . import vision
from . import audio
from . import nlp
from . import graph
from . import time_series
from . import tokenizers
from .tokenizers import (
    CharTokenizer, WordTokenizer, BPETokenizer, BaseTokenizer, load_tokenizer,
)
from . import memory
from .memory import (
    Memory, WorkingMemory, RetrievalMemory, WeightsMemory,
    PermanentMemory,
    TFIDFEmbedder, hash_embedding, random_embedding,
)
from . import bundle
from .bundle import (
    Bundle, save_bundle, load_bundle, inspect_bundle, verify_bundle, diff_bundles,
)
# 15 creative ideas
from . import creative
from .creative import (
    DreamMode, ReplayBuffer, EvolutionaryBreeder, Organism,
    SelfModifyingArchitecture, TimeTravelDebugger, CrossArchTransfer,
    MoralPrinciple, MoralCompass, ModelPersonality, StructuralCausalModel,
    MultiUniverseInference, SyntheticDataGenerator, NLInterpreter,
    MarketModel, ModelMarketplace, Peer, CollaborativeTraining,
    ModelAging, AdversarialPartner,
)
# 2025-era additions: next-gen RAG, agentic AI, reasoning, interpretability, frontier
from .rag import (
    BaselineRAG, GraphRAG, SelfRAG, CorrectiveRAG, HyDE, RAGFusion,
    build_rag, RAG_REGISTRY,
)
from .agents import (
    Tool as AgentTool, ReAct, Reflexion, PlanAndExecute, TreeOfThoughts, SelfAsk,
    Agent, MultiAgentOrchestrator, A2AMessage, A2AProtocol,
)
from .reasoning import (
    ChainOfThought, SelfConsistency, BestOfN, BudgetForcing,
    StepVerifier, ProcessRewardModel,
)
from .interpret import (
    ActivationLogger, ActivationPatcher, CircuitTracer,
    SparseAutoencoder, LogitLens, AttributionPatcher,
)
from .frontier2025 import Mamba2, DeepSeekMoE, TTTModel, Jamba
from .governance.extended import (
    AIRiskAssessment, AuditLogger, ModelCard, ComplianceChecker,
    QuantizationPipeline, PruningPipeline, DistillationPipeline, VibeCoder,
)
# 2025 ecosystem integrations + native memory + distributed training
from .integrations import (
    LangChainAdapter, LlamaIndexAdapter, HaystackAdapter,
    HuggingFaceHub, ModelRegistry,
    FAISSIndex, ChromaStore, MilvusClient, PgvectorStore, VectorStoreRegistry,
    MLflowLogger, WandBLogger,
    LangfuseTracer, ArizeLogger, PhoenixTracer, HeliconeLogger,
    DSPyAdapter, InstructorAdapter, OutlinesAdapter, LMQLAdapter,
)
from .serving import (
    GenerationRequest, GenerationResponse,
    VLLMBackend, SGLangBackend, TGIBackend, OllamaBackend,
    LlamaCppBackend, MLXBackend, LocalAIBackend,
    TensorRTLLMBackend, LMDeployBackend,
    BACKENDS as SERVING_BACKENDS, LLMClient, LLMRouter, OpenAIShim,
)
from .native import (
    CPythonBridge, NativeTensor, SharedMemoryTensor, RingBuffer,
    MemoryArena, CFFIBridge, Pybind11Bridge, SystemMemory,
)
from .distributed import (
    SingleProcessBackend, RayBackend, TorchDistributedBackend,
    DeepSpeedBackend, FSDPBackend, HorovodBackend,
    BACKENDS as DIST_BACKENDS, DistributedTrainer,
)
from .creative.experimental import (
    ModelDreamer, Skill, SkillLibrary, KnowledgeDistiller2,
    ConceptBottleneckModel, MutualInformationEstimator,
    BayesianOptimiser2, MarketplaceV2, CurriculumLearner,
    FederatedPrivacy, NeuroSymbolicLayer,
)
from .creative.experimental2 import (
    ModelReincarnation, SyntheticStudent, PredictiveCache,
    ZeroTrustInference, ModelUnion, ModelPricing, TokenomicsEngine,
    HallucinationDetector, VirtualBody, EmbodiedAgent,
    SkillAdapter, ContinualLearningVault, PromptEvolver,
)
from .creative.experimental3 import (
    ModelTherapist, DataArchaeologist, ModelDreamCatcher,
    AdversarialCoach, ConceptForge, TimeSeriesForecaster2,
    ModelParliament, KnowledgeGraphBuilder, CausalDiscovery,
    NeuralArchitectureSearch2,
)
from .creative.experimental4 import (
    ModelInsurance, EvolvableSkill, SkillEvolution,
    NeuroEvolutionGenome, NeuroEvolution,
    FederatedAuction, Constitution, ModelConstitution,
    QuantumInspiredOptimizer, SyntheticCurriculum, ModelBiography,
    PredictionMarket, NeuralDebateChamber,
)
from .creative.experimental5 import (
    DataTrust, ModelPharmacy, ModelUnion2,
    CrossSpeciesSkill, SkillEvolution2,
    EncapsulatedModel, ModelZoo,
    AdversarialArmsRace,
    GenealogyNode, ModelGenealogy,
    DataValuationMarket,
    ModelEmbryology,
    NeuralSociology,
)
from .creative.llm_enhanced import (
    LLMCallable, LLMEnhancedMoralCompass, LLMEnhancedReflexion,
    LLMEnhancedVibeCoder, LLMEnhancedPromptEvolver,
    LLMEnhancedHallucinationDetector,
)
# Async wrappers (named qai_asyncio to avoid stdlib clash)
from . import asyncio as qai_asyncio
from .asyncio import (
    async_build, async_fit, async_forward, async_predict_many,
    AsyncRAG, AsyncLLMClient, AsyncAgent,
    AsyncBatchProcessor, AsyncExperimentTracker, AsyncDistributedTrainer,
    to_async, run_async,
)
# Benchmarking & performance
# NOTE: `from . import benchmark` would re-bind qai.benchmark to the module,
# overriding the benchmark() function from core.benchmark imported earlier.
# Import the module under a different name so qai.benchmark stays a function.
from . import benchmark as _benchmark_module
from .benchmark import (
    ModelBenchmark, ScalingBenchmark, InferenceComparison,
    LatencyPercentiles, StressTest, RegressionBench,
    quick_benchmark, compare_models,
)
# Re-bind qai.benchmark to the FUNCTION (most users call qai.benchmark(...))
# while the module's classes are still accessible via qai._benchmark_module.*
from .core.benchmark import benchmark
# GPU compute + custom CUDA kernels + device-aware tensors
from . import gpu
from .gpu import (
    DeviceManager, DeviceTensor,
    kernel, thread_id, block_dim, grid_dim, sync_threads,
    matmul as gpu_matmul, softmax as gpu_softmax,
    layernorm as gpu_layernorm, conv2d as gpu_conv2d,
    gather as gpu_gather, scatter as gpu_scatter,
    gpu_memory, empty_cache as gpu_empty_cache,
    memory_pool as gpu_memory_pool, pinned_memory,
    to_gpu, to_cpu, device_count as gpu_device_count,
)
# Real library integrations
from . import integrations_real
from .integrations_real import (
    HuggingFaceModel, HuggingFaceDataset, RealFAISSIndex,
    SentenceTransformerEmbedder, SpacyNLP, SklearnPipeline,
    IntegrationTestRunner,
)
# Documentation utilities
from . import docs
from .docs import (
    get_quickstart, get_advanced_tutorial,
    list_techniques, list_creative_ideas,
    technique_catalog, generate_docs_index,
)
# Type hints + error handling (production hardening)
from . import types as qai_types
from .types import (
    Ok, Err, Result, ok, err,
    QAIError, TechniqueNotFoundError, ModelNotTrainedError,
    BundleCorruptError, BundleSignatureError, GPUUnavailableError,
    OutOfMemoryError, ConvergenceError, DataValidationError, ConfigurationError,
    Forwardable, Fittable, Samplable, Stateful, TokenizerLike,
    retry, safe_call, validate_input, timed, memoize, deprecated,
    require, require_not_none, require_type, require_shape, require_in_range,
    Logger,
)
# Creative ideas 66-75
from .creative.experimental6 import (
    ModelImmuneSystem, NeuroEconomy, CausalBayesianNet,
    ModelRetirementPlan, CognitiveArchitecture, DreamCompiler,
    EpistemicUncertaintyTracker, ModelMutation, ModelMutator,
    CollectiveIntelligence, TimeCrystal,
)
# Creative ideas 76-85
from .creative.experimental7 import (
    KnowledgeEngineer, ModelPlant, ModelGarden, MemoryPalace,
    SkillPyramid, CBTForModels, ModelSandbox, NeuroForensics,
    ModelMarathon, KnowledgeTsunami, ModelHibernation,
)
from .creative.experimental8 import (
    ModelOrganism, ModelStockMarket, WeightArcheology,
    NeuralSymphony, ModelWeather, ModelCourt, DataRefrigerator,
)
from .creative.experimental9 import (
    ModelPrison, NeuralEcosystem, WeightDna,
    ModelHospital, DataMarketplace, ModelPassport, ModelCemetery,
)
from .creative.experimental10 import (
    ModelEmbassy, NeuralCourtship, WeightMuseum,
    ModelGymnast, NeuralBank, ModelTailor,
)
# Creative ideas 109-118 (Phase 4 design additions)
from .creative.experimental11 import (
    ModelAcupuncture, NeuroCryptography, ModelBonsai,
    HallucinationVaccine, QuantumEntanglement, ModelConcert,
    NeuralWeather, WeightGenealogy, ModelShrine, CognitiveEcho,
)
# Creative ideas 119-128 (Phase 5 design additions)
from .creative.experimental12 import (
    ModelTournament, NeuralConstellation, ModelCryptobiology,
    HallucinationAutopsy, WeightSeismograph, ModelEmbassyProtocol,
    NeuralArchaeology, ModelSommelier, PredictionGenealogy, ModelCensus,
)
# Creative ideas 129-138 (Phase 6 design additions)
from .creative.experimental13 import (
    ModelDiplomacy, NeuralGeology, PredictionCourt, WeightTides,
    ModelEconomy, HallucinationCartography, ModelChoir,
    NeuralForensics, ModelCultivation, PredictionArcheology,
)
# Creative ideas 139-148 (Phase 7 design additions)
from .creative.experimental14 import (
    ModelParliament, NeuralMeteorology, WeightGenetics, PredictionTherapy,
    ModelSymphony, NeuralCartography, ModelFuneral, HallucinationEpidemiology,
    WeightTectonics, ModelSpeedReader,
)
# Creative ideas 149-158 (Phase 8 design additions)
from .creative.experimental15 import (
    ModelAstronomy, HallucinationDetective, NeuralGarden, ModelStandUp,
    WeightArchitecture, PredictionForecast, ModelThermostat,
    NeuralLibrary, ModelSymposium, WeightCartography,
)
# Creative ideas 159-168 (Phase 9 design additions)
from .creative.experimental16 import (
    ModelZooCurator, PredictionMuseum, NeuralCourtReporter, ModelArchaeologicalDig,
    WeightAuction, PredictionCartographer, ModelBiographer,
    HallucinationSurgicalTheater, WeightOceanography, ModelConductor,
)
# Creative ideas 169-178 (Phase 10 design additions)
from .creative.experimental17 import (
    ModelConstellation, WeightGeology, PredictionArchaeology, ModelSymposium2,
    NeuralWeatherService, WeightTectonics2, ModelStandUp2, HallucinationDetective2,
    ModelBiography2, PredictionCartographer2,
)
# Federated learning (train across multiple clients with privacy)
from . import federation
from .federation import (
    Client as FedClient, FederatedServer, Aggregator, FedAvgAggregator,
    FedProxAggregator, WeightedAggregator, ClientUpdate, RoundStats, FederationConfig,
)
# Knowledge distillation — THE canonical home is qai.distillation_v2.
# One feature, one entry point: qai.distill(teacher, student, X, y, **params).
# The other classes that used to carry their own copies
# (qai.advanced.KnowledgeDistiller, qai.compression.Distiller,
# qai.KnowledgeDistiller2, qai.ModelDistiller) are thin delegates to it.
from . import distillation_v2
from .distillation_v2 import (
    Distiller, MultiTeacherDistiller, SelfDistiller,
    DistillationConfig, DistillationResult,
    distill, multi_distill, self_distill,
)
# ML experiment versioning (git-for-models)
from . import versioning
from .versioning import VersionRepo, Commit
# Advanced active learning
from . import active_learning_v2
from .active_learning_v2 import (
    QueryStrategy, UncertaintySampling, QueryByCommittee, ExpectedModelChange,
    DensityWeighted, BALD, CoreSet, RandomSampling, HybridStrategy,
    ActiveLearner, LabelOracle, ActiveLearningExperiment,
)
# Neural Architecture Search (auto-discover best architecture)
from . import nas_search
from .nas_search import (
    ArchitectureSpec, TrialResult, NASResult,
    RandomSearch, GridSearch, EvolutionarySearch, ProgressiveGrowth,
)
# Automated data preprocessing pipeline
from . import data_pipeline
from .data_pipeline import (
    ColumnProfile, PipelineStep, DataPipeline,
)
# Model serving gateway (unified REST API)
from . import serving_gateway
from .serving_gateway import Gateway, DeployedModel
# Model observability (production monitoring)
from . import observability
from .observability import (
    MonitoringConfig, MetricSample, PredictionLog, ModelMonitor,
    Alert, AlertRule, DriftDetector, LatencyTracker, HealthChecker,
    MonitoringDashboard,
)
# Model lineage tracker (family tree of model versions)
from . import lineage
from .lineage import (
    LineageNode, LineageEdge, LineageGraph, LineageTracker, LineageVisualizer,
)
# API reference auto-generator (scan all modules, generate markdown/HTML docs)
from . import api_reference
from .api_reference import (
    ApiScanner, ApiEntry, MarkdownGenerator, HtmlGenerator, ApiReference,
)
# Experiment comparison (side-by-side comparison of multiple runs)
from .experiment_compare import (
    ExperimentRun, ExperimentComparison, ComparisonVisualizer, RunCollector,
)
# Collaborative workspace (real-time multi-user editing via WebSocket)
from . import collaborative
from .collaborative import (
    serve_workspace_collab, generate_workspace_collab,
    serve_collab, CollabRoom, CollabPeer, CollabServer,
)
# Plugin marketplace (browse/install/share community plugins)
from . import marketplace
from .marketplace import (
    Plugin, PluginMarketplace,
    search as plugin_search, install as plugin_install,
    uninstall as plugin_uninstall, list_installed as plugin_list_installed,
    list_all as plugin_list_all, categories as plugin_categories,
    load as plugin_load, publish as plugin_publish, stats as plugin_stats,
)
# Model compression advisor (auto-recommend pruning/quantization strategy)
from . import compression_advisor
from .compression_advisor import (
    CompressionRecommendation, CompressionReport, CompressionAdvisor,
    advise as compression_advise, apply as compression_apply,
)
# Model card generator (auto-generate HF-style model cards from metadata)
from .model_card_generator import (
    ModelCardData, ModelCardExtractor, ModelCardTemplate, ModelCardRenderer,
)
# Visual Workflow Builder (chain qai operations as DAG)
from . import workflow
from .workflow import (
    WorkflowNode, WorkflowEdge, WorkflowGraph, WorkflowBuilder,
    OPERATIONS as WORKFLOW_OPERATIONS, TEMPLATES as WORKFLOW_TEMPLATES,
    WorkflowVisualizer, register_operation,
)
# Visual Workspace (the model microscope) -- Phase 4 centerpiece
from . import visual_workspace
from .visual_workspace import (
    serve_workspace, generate_workspace, workspace,
)
# Visual Workspace v2 (model-specific renderers, 3D mode, command palette)
from . import visual_workspace_v2
from .visual_workspace_v2 import (
    serve_workspace as serve_workspace_v2,
    generate_workspace as generate_workspace_v2,
)
# Self-improving models -- Phase 6.4
from . import self_improving
from .self_improving import (
    ContinuousLearner, AutoRetrainer, KnowledgeAccumulator,
    SelfEvaluator, ModelGuardian,
)
# Benchmark suite (synthetic MMLU/HumanEval/GSM8K-style offline eval)
from . import benchmark_suite
from .benchmark_suite import (
    Benchmark, MMLUBenchmark, HumanEvalBenchmark, GSM8KBenchmark,
    GLUEBenchmark, CommonsenseQABenchmark, ARCBenchmark,
    BenchmarkSuite, SuiteReport, Leaderboard,
    run_benchmark as run_benchmark_suite, run_full_suite, list_benchmarks,
    benchmark_info,
)
# Web UI dashboard + tutorials
from . import webui
from .webui import generate_static_html, serve_dashboard
from . import tutorials
from .tutorials import get_tutorial, list_tutorials, generate_tutorial, generate_all as generate_all_tutorials
# The flagship agent system
from . import agent
from .agent import (
    AgentLLM, QuantumAgent, CodeAgent, SelfImprovingAgent, AGISeed,
    create_agent, create_coder, create_self_improving, create_agi,
)
# Model file system (.qmodel with proper extension + shelf + converter + fingerprinter)
from . import modelfile
from .modelfile import (
    ModelFile, ModelShelf, ModelConverter,
    ModelFingerprinter, ModelProvenance, ModelVersioner,
    ModelDiffer, AutoSaver,
    save_trained, load_model_file, inspect_model,
)
# Extended model file features (HF import, registry, signing, linting, migration)
from . import modelfile_extended
from .modelfile_extended import (
    HFModelImporter, HFModelWrapper,
    ModelRegistry2, ModelSigning, ModelLinting,
    ModelMigration, ModelCompatibility,
)
# Weights file system (inspect, edit, merge, visualize, compare, validate)
from . import weights as qai_weights
from .weights import (
    WeightsFile, save_weights, load_weights,
    inspect_weights, inspect_layer, edit_weights,
    merge_weights, visualize_weights, compare_weights,
    validate_weights, WeightsRepository, WeightsEditor,
    WeightsEditWarning,
)
# Per-model logging system
from . import modellog
from .modellog import ModelLog, LoggedModel, ModelLogRegistry
# Universal format registry + converter (18+ ML file formats)
from . import formats
from .formats import (
    FORMATS, FormatInfo, list_formats, format_info,
    is_model_format, is_weights_format, is_safe_format,
    UniversalConverter, convert,
    to_safetensors, to_gguf, to_tflite, detect_format,
)
# Production deployment pipeline
from . import deployment
from .deployment import (
    ModelDeployer, ModelServer, ModelMonitor,
    ABTestRouter, ShadowDeployer, CanaryDeployer,
)
# Weight surgery (precise neuron-level operations)
from . import surgery
from .surgery import (
    WeightSurgeon, NeuronInspector,
    ChannelPruner, WeightProbing,
)
# C bridge extensions (bypass Python limitations)
from . import cbridge
from .cbridge import (
    SourceExtractor, MemoryInspector, FastMath, FastLoops,
    SystemCalls as CBridgedSystemCalls, FileOps as CBridgedFileOps,
    CBridgedMethods,
)
# Resource budget management (RAM/GPU/time/cost)
from . import budget as qai_budget
from .budget import ResourceBudget, ModelSwapper, TimeBudget, CostTracker, ModelSwarm
# Hardware-aware auto-scaling + C-accelerated training
from . import autoscale
from .autoscale import (
    HardwareProfiler as AutoScaleProfiler, AutoConfig, CTrainingLoop, CInferenceEngine,
    TrainingAdvisor, ModelSizingGuide,
)
# AutoML system
from . import automl
from .automl import AutoML, ModelSelector, HyperOptimizer, AutoEnsemble, DataPrepAuto, PipelineBuilder
# Model compression toolkit
from . import compression
from .compression import Quantizer, Pruner, Distiller, CompressionPipeline, CompressionReport
# Structured pruning — THE canonical home is qai.pruning (since v1.8.6).
# One entry point: qai.structured_prune(model, layer_name=..., fraction=...).
# The other entry points (qai.compression.Pruner.structured_prune,
# qai.surgery.ChannelPruner.remove, qai.governance.PruningPipeline.structured_prune)
# are thin delegates to it.
from .pruning import structured_prune, structured_prune_with_finetune
# CLI tool
from . import cli_tool
# HuggingFace converter
from .hf_converter import HFConverter
# Model file extensions (HF importer, registry, inspector, migrator, packager)
from .modelfile_extensions import (
    HuggingFaceImporter, ModelRegistry, ModelInspector,
    ModelMigrator, ModelPackager,
)
# The QuantumMind: flagship super-intelligent agent
from . import mind
from .mind import QuantumMind, CodeAgent, ResearchAgent, CreativeAgent
from . import mcp
from .mcp import (
    MCPServer, Tool, ToolRegistry, tool_registry, create_server,
    FilesystemTool, MemoryTool, GitTool, FetchTool, TimeTool,
    SqliteTool, ExecTool, SequentialThinkingTool,
    CalculatorTool, CodeEvalTool,
    QaiModelTool, QaiLogicTool, QaiMemoryTool, QaiBundleTool,
)
from . import security
from .security import (
    security_scan, prompt_injection_guard, sandbox_run,
    detect_pii, redact_pii, scan_secrets, detect_adversarial_input,
    moderate_content, sanitize_prompt_template,
    APIKeyVault, verify_model_integrity,
)
from . import network
from .network import (
    api, Response, WebSocketClient, oauth, OAuthConnector, orchestrate,
    CircuitBreaker, CircuitBreakerOpenError,
    sign_request, verify_request_signature, RateLimitedClient,
)
from . import advanced
from .advanced import RAGPipeline, KnowledgeDistiller, AIAgent
from . import production
from .production import (
    MultiRoleSelfPlay, Role,
    ComposableSkills, Skill,
    CascadeInference,
    BehaviorContract, BehaviorRule,
    MediaManipulator,
)
from . import infra
from .infra import (
    ModelMerger, StatefulSession, SessionState,
    TrainingStrategies, RiskScorer,
    ModelFamily, Capability,
)
from . import framework
from .framework import (
    OutputRenderer,
    KnowledgeGapTracker, KnowledgeGap,
    IdleTrainer,
    Marketplace, MarketplaceEntry,
    AdvancedVisualizer,
    BidirectionalChannel,
    RouterCore,
    AutomatedScientist,
)
from . import completion
from .completion import (
    Layer, AutoTrainTarget, RewardSystem, WeightToWords,
    MultimodalGenerator, ModelCard, StressTester,
    WeightSafetyToolkit, CICDPipeline, ExperimentDashboard,
)
from . import finalize
from .finalize import (
    OutputControl, OutputControlConfig,
    ExtensibilityGuardrails,
    LazyLoader,
    UniversalManipulator,
    LogicVariantManager,
    TwoTierLogic,
    AutomationLibrary,
    NumericalCorrectness,
    OperationalMaturity,
)
from . import remainder
from .remainder import (
    ShowcaseProject, HardwareProfiler, UniversalHelp,
    NovelInterpretability, ExecutionEnvironment, FreeformOutput,
    FormulaLibrary, LegalCompliance, LogicIntrospection,
    ComposedLogic, APIKeySystem, UserInputPipeline, FeatureWeighting,
)
from . import upgrades
from .upgrades import (
    finetune_model, FederatedCluster, FederatedClient,
    MemorySpanConfig, ContextSpanConfig, configure_memory_span,
    WeightRegionLoader, WeightBlueprint,
    PeerConnection, create_tool_cli, publish_tool,
)

# Top-level math convenience exports
from .math import (
    sigmoid, relu, softmax, gelu, swish, tanh, euclidean_distance, cosine_similarity,
    manhattan_distance, minkowski_distance, huber_loss, focal_loss, triplet_loss,
    kl_divergence, js_divergence, chebyshev_distance, canberra_distance, braycurtis_distance,
    log_sum_exp, safe_divide, levenshtein_distance, jaccard_similarity, contrastive_loss,
    cosine_embedding_loss, margin_ranking_loss, soft_margin_loss, poisson_nll_loss
)
from .system import ResourceGuard, auto_clean_memory, check_ram_threshold, profile_execution_time

# Universal training enhancements (build ON TOP of any existing torch-based
# technique -- NOT new model types). See qai.techniques.training_enhancements.
from .techniques.training_enhancements import (
    SAM, SWA, EMA, Lookahead,
    MixupTrainer, LabelSmoother,
    CosineAnnealer, OneCyclePolicy,
    TTA,
    PseudoLabeler, SnapshotEnsemble,
)

# Standard environment interface + registry (qai.Environment, qai.make_environment, ...)
from .environments import (
    Environment, register_environment, get_environment,
    list_environments, make_environment, BanditEnv, PuzzleWorld,
)

# Generic web dashboard for any model (qai.dashboard)
from .dashboard import dashboard

# One-call evaluation suite (qai.evaluate -> EvaluationReport)
from .evaluate import evaluate, EvaluationReport

# Auto-generated markdown documentation (qai.doc)
from .autodoc import doc

# Active learning loop (qai.ActiveLearningLoop)
from .active_learning import ActiveLearningLoop

# Production drift detection (qai.DriftDetector)
from .drift import DriftDetector, DriftReport

# Live training dashboard (qai.live_training, LiveTrainingCallback)
from .live_training import LiveTrainingCallback, live_training

# Synthetic data generators (beyond make_classification / make_regression)
from .synthetic import make_time_series, make_images, make_text, make_graphs

# Learned meta-compressor + multi-file archive format
# NOTE: qai.compress_bytes is the function; qai.compress is the module.
# (Both are aliased for convenience, but the module wins on attribute access.)
from .compress import (
    compress as compress_bytes, decompress as decompress_bytes,
    MetaCompressor, Archive, ArchiveEntry,
    extract_features as extract_byte_features,
)
# Also expose `qai.compress` as the function (for ergonomics), at the cost
# of shadowing the module. To access the module, use `from qai import compress`.
import qai.compress as _compress_module  # ensure module is importable

# Model profiler (qai.Profiler)
from .profiler import Profiler, Profile as ModelProfile

# Experiment comparison (qai.ExperimentCompare)
from .experiment_compare import (
    ExperimentCompare, ComparisonReport, ComparisonRow,
)
# Note: ExperimentRun is imported from experiment_compare (line ~310), not experiment_tracking
# to avoid the dataclass version with required timestamp field shadowing ours.

# sklearn-compat discoverability layer
from . import sklearn_compat

# Convergent Pattern Compression -- file-native learning system
from .cpc import CPC

# CPC AI -- the complete compression-first intelligence system
from .cpc_ai import (
    CPCAI, CPCBite, KnowledgeGraph, KnowledgeNode,
    NGramPredictor, SymbolicCompressor, Reorganizer,
    VerificationEngine, CuriosityEngine,
    huffman_compress, huffman_decompress,
)

# Model registry (local model marketplace)
from .model_registry import ModelRegistry, RegistryEntry

# Pipeline builder (declarative DAG)
from .pipeline_builder import PipelineBuilder, DAGPipeline, PipelineStep

# Data profiler (dataset health checker)
from .data_profiler import DataProfiler, DataProfile

# A/B testing router (statistical significance)
from .ab_testing import ABTest, ABTestResult

# Model card generator (responsible AI)
from .model_card import ModelCardGenerator, ModelCardContent

# Self-healing model (auto-detect degradation + auto-retrain)
from .self_healing import SelfHealingModel, HealingEvent

# Production model monitor (drift + latency + accuracy over time)
from .model_monitor import ModelMonitor, MonitorReport

# Data augmentor (configurable augmentation pipeline)
from .data_augmentor import DataAugmentor, AugmentationStep

# Federated bridge (combine model pieces from different frameworks)
from .federated_bridge import FederatedBridge, ModelPiece

# Model versioner (track versions, diff, rollback)
from .model_versioner import ModelVersioner, VersionInfo

# Confidence calibrator (Platt scaling + isotonic regression)
from .confidence_calibrator import ConfidenceCalibrator

# Model visualizer (universal architecture diagrams for ALL techniques)
from .model_visualizer import visualize_architecture, visualize_technique, visualize_pipeline

# Model chat (interactive chat window for any model)
from .model_chat import chat_with_model

# Model health checks + token counter + input validation
from .model_health import health_check, token_counter, surprise_score

# Configuration objects for qai.build() — family-specific + cross-cutting
from .configs import (
    MoEConfig, TrainConfig, GenerateConfig,
    TransformerConfig, CNNConfig, RNNConfig, ModernLLMConfig,
    TreeConfig, BoostingConfig, ClusterConfig,
    GNNConfig, VisionConfig, RLConfig, GenerativeConfig,
    RecommenderConfig, SequenceConfig, ProbabilisticConfig,
    get_config_class, config_to_kwargs,
)

# Universal Trainer — grad clip, LR schedule, mixed precision, EMA/SWA, distributed
from .trainer import (
    Trainer, LRScheduler, CosineSchedule, LinearSchedule,
    OneCycleSchedule, PolynomialSchedule, ConstantSchedule, StepSchedule,
    make_optimizer, SCHEDULES,
)

# Parameter-efficient fine-tuning (LoRA, QLoRA, Adapter, IA³, Prefix, Prompt)
from .finetuning import (
    LoRA, QLoRA, Adapter, IA3, PrefixTuning, PromptTuning, apply_finetuning,
)

# Advanced metrics (perplexity, BLEU, ROUGE, FID, CLIP score, ECE, Brier, NDCG)
from .advanced_metrics import (
    perplexity, bleu, rouge, meteor, fid, inception_score, clip_score,
    expected_calibration_error, brier_score, precision_at_k, ndcg_at_k,
    mape, smape, cohens_d,
)

# Alignment trainers (DPO, GRPO, RLHF)
from .alignment import DPOTrainer, GRPOTrainer, RLHFTrainer

# Advanced inference (speculative decoding, streaming, beam search, min-p, typical)
from .inference import (
    stream_generate, speculative_decode, beam_search,
    generate_min_p, generate_typical, best_of_n,
)

# Pretrained weights loading
from .pretrained import load_pretrained, download_weights, list_pretrained

# Dataset loaders (iris, digits, wine, MNIST, CIFAR, GLUE, text corpora)
from .datasets import (
    load_iris, load_digits, load_wine, load_breast_cancer,
    load_diabetes, load_california_housing,
    load_mnist, load_cifar10,
    load_text, load_text_tokens,
    load_glue, load_dataset,
    make_classification_data, make_regression_data, make_blobs_data,
)

# Distributed training (DDP, FSDP)
from .distributed_train import (
    DistributedTrainer, FSDPTrainer, DistributedSampler,
    init_distributed, cleanup_distributed,
    get_world_size, get_rank, is_main_process,
    barrier, all_reduce, all_gather,
)

# Production serving (dynamic batching, streaming SSE, metrics)
from .production_serving import ProductionServer, async_predict

# Quantization (int8, int4, GPTQ, AWQ)
from .quantization import (
    QuantizationConfig, quantize_dynamic, quantize_static,
    quantize_to_int4, GPTQQuantizer, AWQQuantizer,
    benchmark_quantized, model_size_mb,
)

# MLOps pipeline — integrates train → track → register → quantize → serve → monitor
from .mlops import MLOpsPipeline, RunRecord, quick_pipeline

# Fast inference engine (vLLM-style: PagedAttention, continuous batching, prefix caching)
from .fast_inference import FastInferenceEngine, PagedKVCache, PrefixCache, ContinuousBatchingScheduler

# Modern alignment methods (KTO, IPO, SimPO, ORPO, RLVR)
from .modern_alignment import KTOTrainer, IPOTrainer, SimPOTrainer, ORPOTrainer, RLVRTrainer, align

# AI agents (ReAct, tool use, multi-agent, planner)
from .agent_framework import Tool, Toolkit, AgentMemory, Agent, ReActAgent, ToolUseAgent, PlannerAgent, MultiAgent

# Real tokenizers (tiktoken, SentencePiece, HuggingFace)
from .real_tokenizers import RealTokenizer, TiktokenWrapper, SentencePieceWrapper, HuggingFaceTokenizer, AutoTokenizer

# Standard benchmarks (MMLU, HumanEval, GSM8K, MT-Bench, GLUE)
from . import benchmarks

# Test-time scaling (chain-of-thought, self-consistency, tree-of-thoughts, verification)
from .test_time_scaling import ChainOfThought, SelfConsistency, TreeOfThoughts, VerificationGuided, BestOfNWithVerifier, ReasoningChain

# CUDA kernels + hardware detection (auto-select device, FlashAttention, PagedAttention)
from .cuda_kernels import (
    DeviceInfo, HardwareDetector, CudaKernelManager,
    get_hardware_detector, get_device_info, auto_select_device,
    set_device, get_device,
    enable_cuda_kernels, disable_cuda_kernels,
    hardware_report, recommend_device_for_technique,
)

# Model tools (param counting, summary, profiler, hyperparameter suggestions, cost estimator)
from .model_tools import (
    count_parameters, model_summary, suggest_hyperparameters,
    estimate_training_cost, ModelProfiler, model_to_device,
    compare_models, auto_configure,
)

# Precision management (auto-detect bf16/fp16, QAT, dequantization, precision comparison)
from .precision import (
    PrecisionManager, QATTrainer, dequantize,
    set_precision, get_precision, compare_precision,
)

# Creative tools (data augmentation, feature importance, compression advisor, distillation, curriculum)
from .creative_tools import (
    DataAugmentor, FeatureImportanceRanker, CompressionAdvisor,
    ModelDistiller, CurriculumScheduler,
)

# HuggingFace Hub integration (browse, search, download, load, upload)
from .huggingface_hub import (
    browse_models, search_models, list_categories, model_info,
    from_huggingface, load_hf_tokenizer, load_hf_dataset, list_datasets,
    to_huggingface, download_model,
    list_model_zoo, load_from_zoo, MODEL_ZOO,
)

# Auto-ML (try multiple techniques, cross-validate, pick best)
from .auto_ml import (
    suggest_technique, compare_techniques, auto_train,
    AutoMLResult, TechniqueResult,
)

# Pipeline templates (pre-built end-to-end ML workflows)
from .pipeline_templates import (
    PipelineResult, classification_pipeline, regression_pipeline,
    clustering_pipeline, nlp_pipeline, llm_pipeline,
    time_series_pipeline, auto_pipeline, list_pipelines, PIPELINES,
)

# Experiment tracking (MLflow/W&B/JSON integration)
from .experiment_tracking import (
    ExperimentLogger, JSONLogger,
    MLflowLogger, WandBLogger,
    log_experiment, compare_experiments, get_best_experiment,
)

# Smart installer (dependency management, platform detection, doctor, launcher)
from .installer import (
    check_dependencies, install, install_bundle, install_all,
    doctor, detect_platform, platform_info,
    launch_dashboard, launch_chat,
    DEPENDENCIES, BUNDLES,
)

# Streaming training (train on datasets larger than RAM without full download)
from .streaming import (
    stream_dataset, stream_text, search_datasets, browse_datasets,
    StreamingTrainer,
)

# Weight translation (PDF spec: model.weights.translate() — weight-to-concept translator)
from .weight_translation import WeightTranslator

# GGUF/EXL2 quantization formats (added to quantization module)
from .quantization import save_gguf, load_gguf, save_exl2

# Memory vault (tiered persistent memory: hot/warm/cold with smart eviction)
from .memory_vault import MemoryVault, MemoryItem as VaultItem

# Feature store (feature engineering + storage + reuse)
from .feature_store import FeatureStore, FeatureGroup

# Bayesian hyperparameter optimizer (GP + Expected Improvement)
from .hyperparam_optimizer import HyperparameterOptimizer, OptimizationResult

# Agent base class + replay buffer + hyperparameter scheduler
from .agents_base import (
    AgentBase, ReplayBuffer, PrioritizedReplayBuffer, HyperparameterScheduler,
)

# Standard training callbacks (Keras-style)
from .callbacks import (
    Callback, EarlyStopping, CheckpointEvery, LogProgress, TimeLimit, LambdaCallback,
)
from . import callbacks

# Unified model conversion API (qai.export / qai.import_model / qai.conversion.compare)
# Lazy external deps: torch/onnx/skl2onnx/safetensors/etc are imported inside
# each exporter, so importing this module is always fast and never crashes.
from . import conversion
from .conversion import (
    export, import_model, supported_formats, available_backends,
    ConversionReport, compare, benchmark_formats,
    ONNXExporter, TorchScriptExporter, TFLiteExporter, CoreMLExporter,
    TensorRTExporter, OpenVINOExporter, GGUFExporter, SafeTensorsExporter,
    PickleExporter, QAINativeExporter,
)

__all__ = [
    "build", "Model", "register_technique", "Technique", "logic", "LogicError",
    "GridWorld", "get_learning_technique", "LearningTechnique", "CompatibilityError",
    "InputValidationError",
    "Dataset", "StreamingEventConnector", "serve", "build_flask_app", "Pipeline", "cross_validate",
    "k_fold_split", "VotingEnsemble", "AutoTuner", "autotune", "bayesian_optimize", "benchmark",
    "save_model", "load_model", "compress_model", "prune_structured_sparsity", "init_distributed_context",
    "DistributedDataParallelWrapper", "export_to_onnx", "search_architecture", "FederatedServer",
    "federated_averaging", "MultiOutputClassifier", "MultiOutputRegressor", "ClassifierChain",
    "StackingClassifier", "StackingRegressor", "classification_report", "confusion_matrix",
    "mean_absolute_error", "r2_score", "log_loss", "roc_auc_score", "cohen_kappa_score",
    "matthews_corrcoef", "balanced_accuracy_score", "silhouette_score", "davies_bouldin_score",
    "adjusted_rand_score", "normalized_mutual_info_score", "mean_squared_log_error",
    "mean_absolute_percentage_error", "median_absolute_error", "explained_variance_score",
    "max_error", "brier_score", "expected_calibration_error", "precision_recall_curve",
    "roc_curve", "det_curve", "coverage_error", "label_ranking_average_precision_score",
    "zero_one_loss", "hamming_loss", "jaccard_score", "multilabel_confusion_matrix",
    "brier_score_loss", "feature_importance", "permutation_importance", "partial_dependence",
    "shapley_value_approximation", "make_classification", "make_regression", "make_blobs", "get_hardware_info",
    "get_optimal_device", "save_logic_source", "load_logic_source", "StandardScaler", "LabelEncoder",
    "SimpleImputer", "PolynomialFeatures", "MinMaxScaler", "RobustScaler", "Normalizer", "OneHotEncoder",
    "Binarizer", "VarianceThreshold", "SelectKBest", "MaxAbsScaler", "OrdinalEncoder", "TargetEncoder",
    "KBinsDiscretizer", "FunctionTransformer", "ColumnTransformer", "QuantileTransformer",
    "PowerTransformer", "RecursiveFeatureElimination", "SelectFromModel", "clean_dataset", "detect_drift",
    "weight_distance", "calibration_curve", "demographic_parity_difference", "equalized_odds_difference",
    "disparate_impact_ratio", "fairness_audit", "watermark_model", "verify_watermark",
    "ExperimentTracker", "log_telemetry", "sync_tensorboard", "QuantumCircuitSimulator",
    "quantum_expectation", "ReduceLROnPlateau", "data", "system", "math", "runtime",
    "vision", "audio", "nlp", "graph", "time_series",
    "sigmoid", "relu", "softmax", "gelu", "swish", "tanh", "euclidean_distance", "cosine_similarity",
    "manhattan_distance", "minkowski_distance", "huber_loss", "focal_loss", "triplet_loss",
    "kl_divergence", "js_divergence", "chebyshev_distance", "canberra_distance", "braycurtis_distance",
    "log_sum_exp", "safe_divide", "levenshtein_distance", "jaccard_similarity", "contrastive_loss",
    "cosine_embedding_loss", "margin_ranking_loss", "soft_margin_loss", "poisson_nll_loss",
    "ResourceGuard", "auto_clean_memory", "check_ram_threshold", "profile_execution_time",
    # Universal training enhancements (wrap any torch-based qai technique,
    # NOT new model types).
    "SAM", "SWA", "EMA", "Lookahead",
    "MixupTrainer", "LabelSmoother",
    "CosineAnnealer", "OneCyclePolicy",
    "TTA",
    "PseudoLabeler", "SnapshotEnsemble",
    # Standard environment interface + registry
    "Environment", "register_environment", "get_environment",
    "list_environments", "make_environment", "BanditEnv", "PuzzleWorld",
    # Generic web dashboard
    "dashboard",
    # One-call evaluation suite
    "evaluate", "EvaluationReport",
    # Auto-generated documentation
    "doc",
    # Active learning loop
    "ActiveLearningLoop",
    # Drift detection
    "DriftDetector", "DriftReport",
    # Live training dashboard
    "LiveTrainingCallback", "live_training",
    # Synthetic data generators
    "make_time_series", "make_images", "make_text", "make_graphs",
    # Learned meta-compressor + multi-file archive format
    "compress_bytes", "decompress_bytes", "MetaCompressor", "Archive", "ArchiveEntry",
    "extract_byte_features",
    # Model profiler
    "Profiler", "ModelProfile",
    # Experiment comparison
    "ExperimentCompare", "ComparisonReport", "ComparisonRow",
    # Convergent Pattern Compression
    "CPC",
    # CPC AI
    "CPCAI", "CPCBite", "KnowledgeGraph", "KnowledgeNode",
    "NGramPredictor", "SymbolicCompressor", "Reorganizer",
    "VerificationEngine", "CuriosityEngine",
    "huffman_compress", "huffman_decompress",
    # Model registry
    "ModelRegistry", "RegistryEntry",
    # Pipeline builder
    "PipelineBuilder", "DAGPipeline", "PipelineStep",
    # Data profiler
    "DataProfiler", "DataProfile",
    # A/B testing
    "ABTest", "ABTestResult",
    # Model card
    "ModelCardGenerator", "ModelCardContent",
    # Self-healing model
    "SelfHealingModel", "HealingEvent",
    # Model monitor
    "ModelMonitor", "MonitorReport",
    # Data augmentor
    "DataAugmentor", "AugmentationStep",
    # Federated bridge
    "FederatedBridge", "ModelPiece",
    # Model versioner
    "ModelVersioner", "VersionInfo",
    # Confidence calibrator
    "ConfidenceCalibrator",
    # Model visualizer
    "visualize_architecture", "visualize_technique", "visualize_pipeline",
    # Model chat
    "chat_with_model",
    "health_check", "token_counter", "surprise_score",
    # Config objects
    "MoEConfig", "TrainConfig", "GenerateConfig",
    "TransformerConfig", "CNNConfig", "RNNConfig", "ModernLLMConfig",
    "TreeConfig", "BoostingConfig", "ClusterConfig",
    "GNNConfig", "VisionConfig", "RLConfig", "GenerativeConfig",
    "RecommenderConfig", "SequenceConfig", "ProbabilisticConfig",
    "get_config_class", "config_to_kwargs",
    # Trainer
    "Trainer", "LRScheduler", "CosineSchedule", "LinearSchedule",
    "OneCycleSchedule", "PolynomialSchedule", "ConstantSchedule", "StepSchedule",
    "make_optimizer", "SCHEDULES",
    # Fine-tuning
    "LoRA", "QLoRA", "Adapter", "IA3", "PrefixTuning", "PromptTuning",
    "apply_finetuning",
    # Advanced metrics
    "perplexity", "bleu", "rouge", "meteor", "fid", "inception_score",
    "clip_score", "expected_calibration_error", "brier_score",
    "precision_at_k", "ndcg_at_k", "mape", "smape", "cohens_d",
    # Alignment
    "DPOTrainer", "GRPOTrainer", "RLHFTrainer",
    # Advanced inference
    "stream_generate", "speculative_decode", "beam_search",
    "generate_min_p", "generate_typical", "best_of_n",
    # Pretrained
    "load_pretrained", "download_weights", "list_pretrained",
    # Datasets
    "load_iris", "load_digits", "load_wine", "load_breast_cancer",
    "load_diabetes", "load_california_housing",
    "load_mnist", "load_cifar10",
    "load_text", "load_text_tokens",
    "load_glue", "load_dataset",
    "make_classification_data", "make_regression_data", "make_blobs_data",
    # Distributed training
    "DistributedTrainer", "FSDPTrainer", "DistributedSampler",
    "init_distributed", "cleanup_distributed",
    "get_world_size", "get_rank", "is_main_process",
    "barrier", "all_reduce", "all_gather",
    # Production serving
    "ProductionServer", "async_predict",
    # Quantization
    "QuantizationConfig", "quantize_dynamic", "quantize_static",
    "quantize_to_int4", "GPTQQuantizer", "AWQQuantizer",
    "benchmark_quantized", "model_size_mb",
    # MLOps pipeline
    "MLOpsPipeline", "RunRecord", "quick_pipeline",
    # Fast inference
    "FastInferenceEngine", "PagedKVCache", "PrefixCache", "ContinuousBatchingScheduler",
    # Modern alignment
    "KTOTrainer", "IPOTrainer", "SimPOTrainer", "ORPOTrainer", "RLVRTrainer", "align",
    # Agents
    "Tool", "Toolkit", "AgentMemory", "Agent", "ReActAgent", "ToolUseAgent", "PlannerAgent", "MultiAgent",
    # Real tokenizers
    "RealTokenizer", "TiktokenWrapper", "SentencePieceWrapper", "HuggingFaceTokenizer", "AutoTokenizer",
    # Benchmarks
    "benchmarks",
    # Test-time scaling
    "ChainOfThought", "SelfConsistency", "TreeOfThoughts", "VerificationGuided", "BestOfNWithVerifier", "ReasoningChain",
    # CUDA kernels + hardware
    "DeviceInfo", "HardwareDetector", "CudaKernelManager",
    "get_hardware_detector", "get_device_info", "auto_select_device",
    "set_device", "get_device",
    "enable_cuda_kernels", "disable_cuda_kernels",
    "hardware_report", "recommend_device_for_technique",
    # Model tools
    "count_parameters", "model_summary", "suggest_hyperparameters",
    "estimate_training_cost", "ModelProfiler", "model_to_device",
    "compare_models", "auto_configure",
    # GGUF/EXL2
    "save_gguf", "load_gguf", "save_exl2",
    # Memory vault
    "MemoryVault", "VaultItem",
    # Feature store
    "FeatureStore", "FeatureGroup",
    # Bayesian hyperparameter optimizer
    "HyperparameterOptimizer", "OptimizationResult",
    # Agent base class + replay buffer + hyperparameter scheduler
    "AgentBase", "ReplayBuffer", "PrioritizedReplayBuffer", "HyperparameterScheduler",
    # Standard training callbacks
    "Callback", "EarlyStopping", "CheckpointEvery", "LogProgress", "TimeLimit", "LambdaCallback",
    "callbacks",
    # Unified model conversion API
    "conversion", "export", "import_model", "supported_formats", "available_backends",
    "ConversionReport", "compare", "benchmark_formats",
    "ONNXExporter", "TorchScriptExporter", "TFLiteExporter", "CoreMLExporter",
    "TensorRTExporter", "OpenVINOExporter", "GGUFExporter", "SafeTensorsExporter",
    "PickleExporter", "QAINativeExporter",
    # Creative ideas 109-118
    "ModelAcupuncture", "NeuroCryptography", "ModelBonsai",
    "HallucinationVaccine", "QuantumEntanglement", "ModelConcert",
    "NeuralWeather", "WeightGenealogy", "ModelShrine", "CognitiveEcho",
    # Creative ideas 119-128
    "ModelTournament", "NeuralConstellation", "ModelCryptobiology",
    "HallucinationAutopsy", "WeightSeismograph", "ModelEmbassyProtocol",
    "NeuralArchaeology", "ModelSommelier", "PredictionGenealogy", "ModelCensus",
    # Creative ideas 129-138
    "ModelDiplomacy", "NeuralGeology", "PredictionCourt", "WeightTides",
    "ModelEconomy", "HallucinationCartography", "ModelChoir",
    "NeuralForensics", "ModelCultivation", "PredictionArcheology",
    # Creative ideas 139-148
    "ModelParliament", "NeuralMeteorology", "WeightGenetics", "PredictionTherapy",
    "ModelSymphony", "NeuralCartography", "ModelFuneral", "HallucinationEpidemiology",
    "WeightTectonics", "ModelSpeedReader",
    # Creative ideas 149-158
    "ModelAstronomy", "HallucinationDetective", "NeuralGarden", "ModelStandUp",
    "WeightArchitecture", "PredictionForecast", "ModelThermostat",
    "NeuralLibrary", "ModelSymposium", "WeightCartography",
    # Creative ideas 159-168
    "ModelZooCurator", "PredictionMuseum", "NeuralCourtReporter", "ModelArchaeologicalDig",
    "WeightAuction", "PredictionCartographer", "ModelBiographer",
    "HallucinationSurgicalTheater", "WeightOceanography", "ModelConductor",
    # Creative ideas 169-178
    "ModelConstellation", "WeightGeology", "PredictionArchaeology", "ModelSymposium2",
    "NeuralWeatherService", "WeightTectonics2", "ModelStandUp2", "HallucinationDetective2",
    "ModelBiography2", "PredictionCartographer2",
    # Federated learning
    "federation", "FedClient", "FederatedServer", "Aggregator",
    "FedAvgAggregator", "FedProxAggregator", "WeightedAggregator",
    "ClientUpdate", "RoundStats", "FederationConfig",
    # Distillation (canonical: qai.distillation_v2 — one entry point)
    "distillation_v2", "Distiller", "MultiTeacherDistiller", "SelfDistiller",
    "DistillationConfig", "DistillationResult",
    "distill", "multi_distill", "self_distill",
    # Versioning
    "versioning", "VersionRepo", "Commit",
    # Active learning v2
    "active_learning_v2", "QueryStrategy", "UncertaintySampling", "QueryByCommittee",
    "ExpectedModelChange", "DensityWeighted", "BALD", "CoreSet",
    "RandomSampling", "HybridStrategy", "ActiveLearner", "LabelOracle",
    "ActiveLearningExperiment",
    # Neural Architecture Search
    "nas_search", "ArchitectureSpec", "TrialResult", "NASResult",
    "RandomSearch", "GridSearch", "EvolutionarySearch", "ProgressiveGrowth",
    # Data pipeline
    "data_pipeline", "ColumnProfile", "PipelineStep", "DataPipeline",
    # Serving gateway
    "serving_gateway", "Gateway", "DeployedModel",
    # Observability
    "observability", "MonitoringConfig", "MetricSample", "PredictionLog",
    "ModelMonitor", "Alert", "AlertRule", "DriftDetector", "LatencyTracker",
    "HealthChecker", "MonitoringDashboard",
    # Plugin marketplace
    "marketplace", "Plugin", "PluginMarketplace",
    # Compression advisor
    "compression_advisor", "CompressionRecommendation", "CompressionReport",
    "CompressionAdvisor",
    # Structured pruning — canonical: qai.pruning (one entry point)
    "structured_prune", "structured_prune_with_finetune",
    # Model card generator
    "ModelCardData", "ModelCardExtractor", "ModelCardTemplate", "ModelCardRenderer",
    # API reference auto-generator
    "api_reference", "ApiScanner", "ApiEntry", "MarkdownGenerator",
    "HtmlGenerator", "ApiReference",
    # Experiment comparison
    "ExperimentRun", "ExperimentComparison", "ComparisonVisualizer", "RunCollector",
    # Collaborative workspace
    "collaborative", "serve_workspace_collab", "generate_workspace_collab",
    "serve_collab", "CollabRoom", "CollabPeer", "CollabServer",
    # Model lineage tracker
    "lineage", "LineageNode", "LineageEdge", "LineageGraph",
    "LineageTracker", "LineageVisualizer",
    # Visual Workflow Builder
    "workflow", "WorkflowNode", "WorkflowEdge", "WorkflowGraph", "WorkflowBuilder",
    "WorkflowVisualizer", "register_operation",
    # Visual Workspace (model microscope)
    "visual_workspace", "serve_workspace", "generate_workspace", "workspace",
    "visual_workspace_v2", "serve_workspace_v2", "generate_workspace_v2",
    # Self-improving models
    "self_improving", "ContinuousLearner", "AutoRetrainer", "KnowledgeAccumulator",
    "SelfEvaluator", "ModelGuardian",
    # Benchmark suite
    "benchmark_suite", "MMLUBenchmark", "HumanEvalBenchmark", "GSM8KBenchmark",
    "GLUEBenchmark", "CommonsenseQABenchmark", "ARCBenchmark",
    "BenchmarkSuite", "SuiteReport", "Leaderboard", "list_benchmarks",
    "benchmark_info",
]
__version__ = "1.8.7"
