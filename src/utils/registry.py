"""
src/utils/registry.py
"""
from typing import Any, Dict, Optional, Type, TypeVar, Generic

# Define a generic type variable for class types
T = TypeVar("T")

class Registry(Generic[T]):
    """
    A generic registry to map strings to classes (types).
    
    This allows decoupling the configuration strings (in YAML) from 
    the actual Python class implementations.
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): The name of the registry (e.g., 'Model', 'Config').
                        Used for error messaging.
        """
        self._name: str = name
        self._module_dict: Dict[str, Type[T]] = {}

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    def __repr__(self) -> str:
        return f"{self._name} Registry(size={len(self)}, items={list(self._module_dict.keys())})"

    def register(self, name: Optional[str] = None):
        """
        Decorator to register a class.

        Args:
            name (str, optional): The key to register the class under. 
                                  If None, uses the class name (cls.__name__).
        
        Usage:
            @registry.register("my_model")
            class MyModel: ...
        """
        def _register_wrapper(cls: Type[T]) -> Type[T]:
            # Determine the key
            key = name if name is not None else cls.__name__
            
            # 1. Duplicate check: prevent overwriting existing keys silently
            if key in self._module_dict:
                raise ValueError(
                    f"Key '{key}' is already registered in {self._name} Registry! "
                    f"Existing: {self._module_dict[key]}, New: {cls}"
                )
            
            # 2. Registration
            self._module_dict[key] = cls
            return cls
        
        return _register_wrapper

    def get(self, key: str) -> Type[T]:
        """
        Retrieve a class by its registered key.

        Args:
            key (str): The name of the module to retrieve.

        Returns:
            Type[T]: The class type (not an instance).
        
        Raises:
            KeyError: If the key is not found.
        """
        if key not in self._module_dict:
            available = list(self._module_dict.keys())
            raise KeyError(
                f"'{key}' is not found in {self._name} Registry. "
                f"Available keys: {available}"
            )
        return self._module_dict[key]

# --- Global Registry Instances ---

# Registry for Model Architectures (e.g., DeepWindModel, DeepWindV2)
# Expected type: PreTrainedModel or nn.Module
MODEL_REGISTRY = Registry("Model")

# Registry for Configuration Classes (e.g., DeepWindConfig)
# Expected type: PretrainedConfig
CONFIG_REGISTRY = Registry("Config")

# Registry for Datasets (e.g., DeepWindTrainDataset)
# Expected type: Dataset or IterableDataset
DATASET_REGISTRY = Registry("Dataset")


# --- Public Decorators (Syntactic Sugar) ---

def register_model(name: Optional[str] = None):
    return MODEL_REGISTRY.register(name)

def register_config(name: Optional[str] = None):
    return CONFIG_REGISTRY.register(name)

def register_dataset(name: Optional[str] = None):
    return DATASET_REGISTRY.register(name)

