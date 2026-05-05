
import torch

CAR_CLASSES = {2: "car", 3: "moto", 5: "bus", 7: "truck", 0: "person"}
CONF_CARS   = 0.40


def run_cars(model, frame):
    """
        Ruleaza inferenta pentru detectia de vehicule pe un frame video.

        Utilizeaza YOLOv11m pentru a detecta  clasele definite ca vehicule si pietoni.

        Args:
            model: Model YOLO incarcat.
            frame: Imaginea (frame-ul video) pe care se face inferenta.

        Returns:
            Rezultatul primei predictii YOLO (frame curent).
    """
    with torch.no_grad():
        return model(
            frame,
            verbose=False,
            conf=CONF_CARS,
            classes=list(CAR_CLASSES.keys()),
            imgsz=640,
            iou=0.45,
            agnostic_nms=True,
        )[0]

def collect_car_data(results) -> list:
    """
       Extrage si formateaza datele despre vehicule si pietoni din rezultatele YOLO.

       Primeste output-ul brut al modelului de detectie si returneaza o lista
       simplificata de dictionare, continand doar vehiculele detectate.

       Returns:
           list: Lista de obiecte detectate, fiecare avand:
               - label: numele clasei (ex: car, truck)
               - conf: scorul de incredere
               - box: coordonatele bounding box [x1, y1, x2, y2]
       """

    out = []
    if results is None or results.boxes is None:
        return out
    for box, cls, conf in zip(
        results.boxes.xyxy.cpu().numpy(),
        results.boxes.cls.cpu().numpy(),
        results.boxes.conf.cpu().numpy(),
    ):

        #adaugare in lista obiectelor detectate, doar daca fac parte din CAR_CLASSES
        cls_int = int(cls)
        if cls_int not in CAR_CLASSES:
            continue
        out.append({
            "label": CAR_CLASSES[cls_int],
            "conf":  round(float(conf), 3),
            "box":   [int(x) for x in box],
        })
    return out
