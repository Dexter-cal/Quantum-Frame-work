"""
qai.creative -- the 15 creative ideas built on top of the Quantum framework.

These are experimental, opinionated, "wouldn't it be cool if..." modules
that go beyond the standard ML techniques catalog. They are designed to
compose cleanly with everything else in qai.

Modules
-------
Idea 1   dream.DreamMode                -- idle-time imagination + replay
Idea 2   evolution.EvolutionaryBreeder  -- population-based model breeding
Idea 3   self_modifying.SelfModifyingArchitecture -- model edits its own topology
Idea 4   debugging.TimeTravelDebugger   -- scrub through training history
Idea 5   alignment.MoralCompass         -- constitutional principle guardrails
Idea 6   alignment.ModelPersonality     -- OCEAN traits for sampling behavior
Idea 7   debugging.CrossArchTransfer    -- distill across architecture families
Idea 8   alignment.StructuralCausalModel -- do-calculus + counterfactuals
Idea 9   multiverse.MultiUniverseInference -- K parallel counterfactual rollouts
Idea 10  multiverse.SyntheticDataGenerator -- unified multi-modal synth API
Idea 11  multiverse.NLInterpreter       -- one-sentence NL explanations
Idea 12  ecosystem.ModelMarketplace     -- publish / discover / rate models
Idea 13  ecosystem.CollaborativeTraining -- federated + gossip training
Idea 14  ecosystem.ModelAging           -- shelf life + staleness tracking
Idea 15  ecosystem.AdversarialPartner   -- auto-red-team + hard-negative mining

Idea 16-25 (experimental.py):
Idea 16  ModelDreamer                   -- inputs that maximise a target neuron
Idea 17  SkillLibrary                   -- composable skill chains + planner
Idea 18  KnowledgeDistiller2            -- multi-level KD (output + attn + hidden)
Idea 19  ConceptBottleneckModel         -- explicit human-interpretable concept layer
Idea 20  MutualInformationEstimator     -- privacy / OOD auditing
Idea 21  BayesianOptimiser2             -- high-dim BO with trust regions
Idea 22  MarketplaceV2                  -- royalties + staking + revenue settlement
Idea 23  CurriculumLearner              -- difficulty-aware sampler + prereq discovery
Idea 24  FederatedPrivacy               -- DP-SGD + secure aggregation
Idea 25  NeuroSymbolicLayer             -- differentiable Horn-clause reasoning
"""
from __future__ import annotations

from .dream import DreamMode, ReplayBuffer
from .evolution import EvolutionaryBreeder, Organism
from .self_modifying import SelfModifyingArchitecture
from .debugging import TimeTravelDebugger, CrossArchTransfer
from .alignment import (
    MoralPrinciple, MoralCompass, ModelPersonality, StructuralCausalModel,
)
from .multiverse import (
    MultiUniverseInference, SyntheticDataGenerator, NLInterpreter,
)
from .ecosystem import (
    MarketModel, ModelMarketplace, Peer, CollaborativeTraining,
    ModelAging, AdversarialPartner,
)
from .experimental import (
    ModelDreamer, Skill, SkillLibrary, KnowledgeDistiller2,
    ConceptBottleneckModel, MutualInformationEstimator,
    BayesianOptimiser2, MarketplaceV2, CurriculumLearner,
    FederatedPrivacy, NeuroSymbolicLayer,
)
from .experimental2 import (
    ModelReincarnation, SyntheticStudent, PredictiveCache,
    ZeroTrustInference, ModelUnion, ModelPricing, TokenomicsEngine,
    HallucinationDetector, VirtualBody, EmbodiedAgent,
    SkillAdapter, ContinualLearningVault, PromptEvolver,
)
from .experimental3 import (
    ModelTherapist, DataArchaeologist, ModelDreamCatcher,
    AdversarialCoach, ConceptForge, TimeSeriesForecaster2,
    ModelParliament, KnowledgeGraphBuilder, CausalDiscovery,
    NeuralArchitectureSearch2,
)
from .experimental4 import (
    ModelInsurance, EvolvableSkill, SkillEvolution,
    NeuroEvolutionGenome, NeuroEvolution,
    FederatedAuction, Constitution, ModelConstitution,
    QuantumInspiredOptimizer, SyntheticCurriculum, ModelBiography,
    PredictionMarket, NeuralDebateChamber,
)
from .experimental5 import (
    DataTrust, ModelPharmacy, ModelUnion2,
    CrossSpeciesSkill, SkillEvolution2,
    EncapsulatedModel, ModelZoo,
    AdversarialArmsRace,
    GenealogyNode, ModelGenealogy,
    DataValuationMarket,
    ModelEmbryology,
    NeuralSociology,
)
from .experimental6 import (
    ModelImmuneSystem, NeuroEconomy, CausalBayesianNet,
    ModelRetirementPlan, CognitiveArchitecture, DreamCompiler,
    EpistemicUncertaintyTracker, ModelMutation, ModelMutator,
    CollectiveIntelligence, TimeCrystal,
)
from .experimental7 import (
    KnowledgeEngineer, ModelPlant, ModelGarden, MemoryPalace,
    SkillPyramid, CBTForModels, ModelSandbox, NeuroForensics,
    ModelMarathon, KnowledgeTsunami, ModelHibernation,
)
from .experimental8 import (
    ModelOrganism, ModelStockMarket, WeightArcheology,
    NeuralSymphony, ModelWeather, ModelCourt, DataRefrigerator,
)
from .experimental9 import (
    ModelPrison, NeuralEcosystem, WeightDna,
    ModelHospital, DataMarketplace, ModelPassport, ModelCemetery,
)
from .experimental10 import (
    ModelEmbassy, NeuralCourtship, WeightMuseum,
    ModelGymnast, NeuralBank, ModelTailor,
)
from .llm_enhanced import (
    LLMCallable, LLMEnhancedMoralCompass, LLMEnhancedReflexion,
    LLMEnhancedVibeCoder, LLMEnhancedPromptEvolver,
    LLMEnhancedHallucinationDetector,
)

__all__ = [
    # Idea 1
    "DreamMode", "ReplayBuffer",
    # Idea 2
    "EvolutionaryBreeder", "Organism",
    # Idea 3
    "SelfModifyingArchitecture",
    # Idea 4
    "TimeTravelDebugger",
    # Idea 5
    "MoralPrinciple", "MoralCompass",
    # Idea 6
    "ModelPersonality",
    # Idea 7
    "CrossArchTransfer",
    # Idea 8
    "StructuralCausalModel",
    # Idea 9
    "MultiUniverseInference",
    # Idea 10
    "SyntheticDataGenerator",
    # Idea 11
    "NLInterpreter",
    # Idea 12
    "MarketModel", "ModelMarketplace",
    # Idea 13
    "Peer", "CollaborativeTraining",
    # Idea 14
    "ModelAging",
    # Idea 15
    "AdversarialPartner",
    # Idea 16-25 (experimental)
    "ModelDreamer", "Skill", "SkillLibrary", "KnowledgeDistiller2",
    "ConceptBottleneckModel", "MutualInformationEstimator",
    "BayesianOptimiser2", "MarketplaceV2", "CurriculumLearner",
    "FederatedPrivacy", "NeuroSymbolicLayer",
    # Idea 26-35 (experimental2)
    "ModelReincarnation", "SyntheticStudent", "PredictiveCache",
    "ZeroTrustInference", "ModelUnion", "ModelPricing", "TokenomicsEngine",
    "HallucinationDetector", "VirtualBody", "EmbodiedAgent",
    "SkillAdapter", "ContinualLearningVault", "PromptEvolver",
    # Idea 36-45 (experimental3)
    "ModelTherapist", "DataArchaeologist", "ModelDreamCatcher",
    "AdversarialCoach", "ConceptForge", "TimeSeriesForecaster2",
    "ModelParliament", "KnowledgeGraphBuilder", "CausalDiscovery",
    "NeuralArchitectureSearch2",
    # Idea 46-55 (experimental4)
    "ModelInsurance", "EvolvableSkill", "SkillEvolution",
    "NeuroEvolutionGenome", "NeuroEvolution",
    "FederatedAuction", "Constitution", "ModelConstitution",
    "QuantumInspiredOptimizer", "SyntheticCurriculum", "ModelBiography",
    "PredictionMarket", "NeuralDebateChamber",
    # Idea 56-65 (experimental5)
    "DataTrust", "ModelPharmacy", "ModelUnion2",
    "CrossSpeciesSkill", "SkillEvolution2",
    "EncapsulatedModel", "ModelZoo",
    "AdversarialArmsRace",
    "GenealogyNode", "ModelGenealogy",
    "DataValuationMarket",
    "ModelEmbryology",
    "NeuralSociology",
    # Idea 66-75 (experimental6)
    "ModelImmuneSystem", "NeuroEconomy", "CausalBayesianNet",
    "ModelRetirementPlan", "CognitiveArchitecture", "DreamCompiler",
    "EpistemicUncertaintyTracker", "ModelMutation", "ModelMutator",
    "CollectiveIntelligence", "TimeCrystal",
    # Idea 76-85 (experimental7)
    "KnowledgeEngineer", "ModelPlant", "ModelGarden", "MemoryPalace",
    "SkillPyramid", "CBTForModels", "ModelSandbox", "NeuroForensics",
    "ModelMarathon", "KnowledgeTsunami", "ModelHibernation",
    # Idea 86-95 (experimental8)
    "ModelOrganism", "ModelStockMarket", "WeightArcheology",
    "NeuralSymphony", "ModelWeather", "ModelCourt", "DataRefrigerator",
    # Idea 93-100 (experimental9)
    "ModelPrison", "NeuralEcosystem", "WeightDna",
    "ModelHospital", "DataMarketplace", "ModelPassport", "ModelCemetery",
    # Idea 101-108 (experimental10)
    "ModelEmbassy", "NeuralCourtship", "WeightMuseum",
    "ModelGymnast", "NeuralBank", "ModelTailor",
    # LLM-enhanced versions
    "LLMCallable", "LLMEnhancedMoralCompass", "LLMEnhancedReflexion",
    "LLMEnhancedVibeCoder", "LLMEnhancedPromptEvolver",
    "LLMEnhancedHallucinationDetector",
]
