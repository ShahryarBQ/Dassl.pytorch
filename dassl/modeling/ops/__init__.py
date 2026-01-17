from .mmd import MaximumMeanDiscrepancy
from .conv import *
from .dsbn import DSBN1d, DSBN2d
from .mixup import mixup
from .efdmix import (
    EFDMix, random_efdmix, activate_efdmix, run_with_efdmix, deactivate_efdmix,
    crossdomain_efdmix, run_without_efdmix
)
from .mixstyle import (
    MixStyle, random_mixstyle, activate_mixstyle, run_with_mixstyle,
    deactivate_mixstyle, crossdomain_mixstyle, run_without_mixstyle,
    sots_mixstyle, MixStyle_dfed, MixStyle_Sigma_dfed,    # DDG-added
    MixStyle_ViT
)
from .attention import *
from .transnorm import TransNorm1d, TransNorm2d
from .sequential2 import Sequential2
from .reverse_grad import ReverseGrad
from .cross_entropy import cross_entropy
from .optimal_transport import SinkhornDivergence, MinibatchEnergyDistance
# Author-added
from .dsu import (
    DistributionUncertainty, DistributionUncertainty_dfed, DistributionUncertainty_Sigma_dfed,
    DistributionUncertainty_ViT
)
from .oma import (
    OMA,  random_oma, activate_oma, run_with_oma,
    deactivate_oma, crossdomain_oma, run_without_oma,
    OMA_dfed, OMA_Sigma_dfed
)
from .styleexplore import (
    StyleExplore, random_styleexplore, activate_styleexplore, run_with_styleexplore,
    deactivate_styleexplore, crossdomain_styleexplore, run_without_styleexplore,
    StyleExplore_Sigma_dfed,    # DDG-added
)
