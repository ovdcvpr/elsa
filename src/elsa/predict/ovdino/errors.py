class DetrixImportError(ImportError):
    def __init__(self):
        message = """
        ```
        cd src/elsa/predict/ovdino
        ```
        Then, run the following commands:
        ```
        git clone https://github.com/IDEA-Research/detrex.git
        cd detrex
        git submodule init
        git submodule update
        python -m pip install -e detectron2
        pip install -e .
        ```
        """
        super().__init__(message)


class DetectronImportError(ImportError):
    def __init__(self):
        message = (
            "Please install Detectron2 using the following command:\n"
            "`python -m pip install -e detectron2`"
        )
        super().__init__(message)
