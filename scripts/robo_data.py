# from roboflow import Roboflow
# rf = Roboflow(api_key="ldw81dxCaW2k0rbUMoUq")
# project = rf.workspace("fieldvision").project("soccer-1zdbh")
# version = project.version(1)
# dataset = version.download("yolov8")
                


from roboflow import Roboflow
rf = Roboflow(api_key="ldw81dxCaW2k0rbUMoUq")
project = rf.workspace("smart-football-object-detection").project("smart-football-object-detection")
version = project.version(8)
dataset = version.download("yolov8")
                