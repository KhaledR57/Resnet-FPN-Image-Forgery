import cv2
import json
from pathlib import Path
from multiprocessing import Pool
from pycocotools.coco import COCO

coco = COCO("annotations/instances_val2017.json") #annotations.json

def inpaintImage(imageID, annotationID):
    imageName = coco.loadImgs(int(imageID))[0]["file_name"]
    image = cv2.imread(str(Path("images") / imageName))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    annotation = coco.loadAnns(annotationID)[0]
    mask = coco.annToMask(annotation)

    inpaintedNavierStokesImage = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)

    inpaintedNavierStokesImage = cv2.cvtColor(inpaintedNavierStokesImage, cv2.COLOR_YUV2BGR)

    inpainetedImagesDirectory = Path("inpaintedImages")
    inpainetedImagesDirectory.mkdir(exist_ok = True)

    inpaintedNavierStokesImageName = "navier_stokes_" + imageName.replace("jpg", "png")
    inpaintedNavierStokesImagePath = inpainetedImagesDirectory / inpaintedNavierStokesImageName

    cv2.imwrite(str(inpaintedNavierStokesImagePath), inpaintedNavierStokesImage)

with open("annotationsIDs.json", "r") as annotationIDs:
    with Pool() as pool:
        pool.starmap(inpaintImage, json.load(annotationIDs).items())
