"""
Architectural DNA Module
Defines the genetic markers and evolutionary characteristics of system modules.
Each module carries performance characteristics, dependency graphs, and failure modes.
"""
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ModuleType(Enum):
    """Types of modules in the ecosystem"""
    PROCESSOR = "processor"
    ANALYZER = "analyzer"
    OPTIMIZER = "optimizer"
    INTEGRATOR = "integrator"
    MONITOR = "monitor"
    EVOLVER = "evolver"

class PerformanceTrait(Enum):
    """Genetic performance traits"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"
    RESOURCE_EFFICIENCY = "resource_efficiency"

@dataclass
class DependencyNode:
    """Represents a dependency in the module graph"""
    module_id: str
    dependency_type: str
    strength: float = 1.0  # 0.0 to 1.0
    critical: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "module_id": self.module_id,
            "dependency_type": self.dependency_type,
            "strength": self.strength,
            "critical": self.critical
        }

@dataclass
class FailureMode:
    """Records failure characteristics for evolutionary learning"""
    failure_type: str
    frequency: float = 0.0
    severity: float = 0.0  # 0.0 to 1.0
    recovery_time: float = 0.0
    last_occurred: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "failure_type": self.failure_type,
            "frequency": self.frequency,
            "severity": self.severity,
            "recovery_time": self.recovery_time,
            "last_occurred": self.last_occurred.isoformat() if self.last_occurred else None
        }

@dataclass
class ArchitecturalDNA:
    """
    Genetic blueprint for a system module.
    Contains performance characteristics, dependencies, and evolutionary markers.
    """
    module_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    module_type: ModuleType = ModuleType.PROCESSOR
    version: str = "1.0.0"
    
    # Performance characteristics (genetic markers)
    performance_traits: Dict[PerformanceTrait, float] = field(default_factory=dict)
    
    # Dependency graph
    dependencies: List[DependencyNode] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    
    # Failure modes and recovery patterns
    failure_modes: Dict[str, FailureMode] = field(default_factory=dict)
    
    # Evolutionary markers
    generation: int = 1
    mutation_count: int = 0
    fitness_score: float = 0.0
    last_evaluated: datetime = field(default_factory=datetime.utcnow)
    
    # Configuration and state
    configuration: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default performance traits if none provided"""
        if not self.performance_traits:
            self.performance_traits = {
                PerformanceTrait.LATENCY: 0.5,
                PerformanceTrait.THROUGHPUT: 0.5,
                PerformanceTrait.ACCURACY: 0.5,
                PerformanceTrait.RELIABILITY: 0.5,
                PerformanceTrait.SCALABILITY: 0.5,
                PerformanceTrait.RESOURCE_EFFICIENCY: 0.5
            }
    
    def add_dependency(self, module_id: str, dependency_type: str, strength: float = 1.0, critical: bool = False) -> None:
        """Add a dependency to this module"""
        dependency = DependencyNode(
            module_id=module_id,
            dependency_type=dependency_type,
            strength=strength,
            critical=critical
        )
        self.dependencies.append(dependency)
        logger.info(f"Added dependency: {module_id} to {self.module_id}")
    
    def record_failure(self, failure_type: str, severity: float, recovery_time: float) -> None:
        """Record a failure mode for evolutionary learning"""
        if failure_type not in self.failure_modes:
            self.failure_modes[failure_type] = FailureMode(
                failure_type=failure_type,
                frequency=0.0,
                severity=0.0,
                recovery_time=0.0
            )
        
        failure = self.failure_modes[failure_type]
        failure.frequency += 1
        failure.severity = (failure.severity + severity) / 2  # Moving average
        failure.recovery_time = (failure.recovery_time + recovery_time) / 2
        failure.last_occurred = datetime.utcnow()
        
        logger.warning(f"Recorded failure: {failure_type} for module {self.module_id}")
    
    def mutate_trait(self, trait: PerformanceTrait, delta: float) -> None:
        """Mutate a performance trait (evolutionary pressure)"""
        if trait in self.performance_traits:
            current = self.performance_traits[trait]
            new_value = max(0.0, min(1.0, current + delta))
            self.performance_traits[trait] = new_value
            self.mutation_count += 1
            logger.info(f"Mutated {trait.value}: {current:.3f} -> {new_value:.3f}")
    
    def calculate_fitness(self) -> float:
        """Calculate overall fitness score based on performance traits"""
        weights = {
            PerformanceTrait.LATENCY: 0.15,
            PerformanceTrait.THROUGHPUT: 0.20,
            PerformanceTrait.ACCURACY: 0.25,
            PerformanceTrait.RELIABILITY: 0.20,
            PerformanceTrait.SCALABILITY: 0.10,
            PerformanceTrait.RESOURCE_EFFICIENCY: 0.10
        }
        
        fitness = 0.0
        for trait, weight in weights.items():
            if trait in self.performance_traits:
                fitness += self.performance_traits[trait] * weight
        
        # Penalize for frequent severe failures
        failure_penalty = 0.0
        for failure in self.failure_modes.values():
            if failure.severity > 0.7 and failure.frequency > 3:
                failure_penalty += 0.1
        
        self.fitness_score = max(0.0, fitness - failure_penalty)
        self.last_evaluated = datetime.utcnow()
        
        logger.info(f"Calculated fitness for {self.module_id}: {self.fitness_score:.3f}")
        return self.fitness_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['module_type'] = self.module_type.value
        data['performance_traits'] = {k.value: v for k, v in self.performance_traits.items()}
        data['dependencies'] = [d.to_dict() for d in self.dependencies]
        data['failure_modes'] = {k: v.to_dict() for k, v in self.failure_modes.items()}
        data['last_evaluated'] = self.last_evaluated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArchitecturalDNA':
        """Create from dictionary"""
        # Handle enum conversions
        data['module_type'] = ModuleType(data['module_type'])
        
        # Convert performance traits back to enum keys
        if 'performance_traits' in data:
            perf_data = data['performance_traits']
            data['performance_traits'] = {PerformanceTrait(k): v for k, v in perf_data.items()}
        
        # Reconstruct dependencies
        if 'dependencies' in data:
            deps_data = data['dependencies']
            data['dependencies'] = [
                DependencyNode(
                    module_id=d['module_id'],
                    dependency_type=d['dependency_type'],
                    strength=d.get('strength', 1.0),
                    critical=d.get('critical', False)
                ) for d in deps_data