class DetectronImportError(ImportError):
    def __init__(self):
        message = (
            "Please install Detectron2 using the following command:\n"
            "python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'"
        )
        super().__init__(message)


class DeticImportError(ImportError):
    def __init__(self):
        message = """
        You must install Detic. Please navigate to the Detic folder depending on where you are:
        ```
        pip install -e 'git+https://github.com/nateagr/Detic.git@main#egg=detic-fork&submodules=true'
        ```
        """

        super().__init__(message)

class CenterNetImportError(ImportError):
    def __init__(self):
        message = """
        You must add this prefix to the centernet lines in your Detic repo:
        ```
        from centernet.modeling.backbone.fpn_p5 import LastLevelP6P7_P5
        from centernet.modeling.backbone.bifpn import BiFPN
        
        ```
        becomes
        ```
        from Detic.third_party.CenterNet2.centernet.modeling.backbone.fpn_p5 import LastLevelP6P7_P5
        from Detic.third_party.CenterNet2.centernet.modeling.backbone.bifpn import BiFPN
        ```
        """
        super().__init__(message)
