class YoloImportError(ImportError):
    def __init__(self):
        message = """
        ```
        git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
        cd YOLO-World
        pip install torch wheel -q
        pip install -e .
        """
        super().__init__(message)

