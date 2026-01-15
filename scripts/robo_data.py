from roboflow import Roboflow
rf = Roboflow(api_key="ldw81dxCaW2k0rbUMoUq")
# project = rf.workspace("fieldvision").project("soccer-1zdbh")
# version = project.version(1)
# dataset = version.download("yolov8")
                
# project = rf.workspace("footballanalysis-dodjc").project("football-player-detection-5ncad")
# version = project.version(4)
# dataset = version.download("yolov8")

# project = rf.workspace("minaehyeon").project("football-player-jrjtj")
# version = project.version(5)
# dataset = version.download("yolov8")

project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(14)
dataset = version.download("yolov8")