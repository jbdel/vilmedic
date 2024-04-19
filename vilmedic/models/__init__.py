try:
    import pkg_resources

    # pkg_resources.require("transformers==4.23.1")
    from vilmedic.models.mvqa.MVQA import MVQA

    # from vilmedic.models.rrs.RRS import RRS
    # from vilmedic.models.rrs.RRS_SCST import RRS_SCST
    from vilmedic.models.rrs.RRS_HF import RRS_HF

    from vilmedic.models.rrg.RRG_HF import RRG_HF
    from vilmedic.models.rrg.RRG import RRG
    # from vilmedic.models.rrg.RRG_FORCE import RRG_FORCE
    # from vilmedic.models.rrg.RRG_SCST import RRG_SCST

    from vilmedic.models.selfsup.conVIRT import ConVIRT
    from vilmedic.models.selfsup.GLoRIA import GLoRIA
except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
    pass

# try:
#     import pkg_resources
#
#     pkg_resources.require("transformers==4.25.1")
#     from vilmedic.models.selfsup.VQModel import VQModel
# except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
#     pass
