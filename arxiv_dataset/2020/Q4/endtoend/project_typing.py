from typing import Dict, List, Tuple, Set, NewType, Optional, FrozenSet

Feature = NewType("Feature", str)
Value = NewType("Value", str)
FeatureValue = Tuple[Feature, Value]
Agent = Dict[Feature, Value]
Panel = FrozenSet[int]
