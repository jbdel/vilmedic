try:
    import pkg_resources

    pkg_resources.require("transformers==4.23.1")
    from vilmedic.models.mvqa.MVQA import MVQA

    from vilmedic.models.rrs.RRS import RRS
    from vilmedic.models.rrs.RRS_SCST import RRS_SCST
    from vilmedic.models.rrs.SumHugMono_SCST import SumHugMono_SCST

    from vilmedic.models.selfsup.clip.VAE import VAE
    from vilmedic.models.selfsup.clip.DALLE import DALLE
    from vilmedic.models.selfsup.clip.CLIP import CLIP

    from vilmedic.models.rrg.RRG import RRG
    from vilmedic.models.rrg.RRG_FORCE import RRG_FORCE
    from vilmedic.models.rrg.RRG_MULTI import RRG_MULTI
    from vilmedic.models.rrg.RRG_EEG import RRG_EEG
    from vilmedic.models.rrg.RRG_PPO import RRG_PPO
    from vilmedic.models.rrg.RRG_SCST import RRG_SCST

    from vilmedic.models.selfsup.conVIRT import ConVIRT
    from vilmedic.models.selfsup.SimCLR import SimCLR
    from vilmedic.models.selfsup.GLoRIA import GLoRIA
    from vilmedic.models.selfsup.VICREG import VICREG
    from vilmedic.models.selfsup.SwaV import SwaV
    from vilmedic.models.selfsup.PCL import PCL
    from vilmedic.models.selfsup.PCL2 import PCL2
except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
    pass

try:
    import pkg_resources

    pkg_resources.require("transformers==4.25.1")
    from vilmedic.models.selfsup.VQModel import VQModel
except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
    pass
