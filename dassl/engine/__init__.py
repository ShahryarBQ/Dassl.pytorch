from .build import TRAINER_REGISTRY, build_trainer  # isort:skip
from .trainer import TrainerX, TrainerXU, TrainerBase, SimpleTrainer, SimpleNet  # isort:skip
# DDG-added
from .trainer import TrainerX_dfed, SimpleTrainer_dfed

from .da import *
from .dg import *
from .ssl import *
