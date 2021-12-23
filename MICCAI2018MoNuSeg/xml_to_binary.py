from PIL import Image,ImageDraw
import xml.dom.minidom
import numpy as np
def polygon2mask_default(w: int, h: int, polygons: list) -> Image.Image:
    '''

    '''
    binary_mask = Image.new('L', (w, h), 0)
    for polygon in polygons:
        ImageDraw.Draw(binary_mask).polygon(polygon, outline=255, fill=255)
    return binary_mask

def xml_to_binary_mask(w, h, filename: str, polygon2mask=polygon2mask_default) -> Image.Image:
    xml_file = filename
    xDoc = xml.dom.minidom.parse(xml_file).documentElement
    Regions = xDoc.getElementsByTagName('Region')
    xy = []
    for i, Region in enumerate(Regions):
        verticies = Region.getElementsByTagName('Vertex')
        xy.append(np.zeros((len(verticies), 2)))
        for j, vertex in enumerate(verticies):
            xy[i][j][0], xy[i][j][1] = float(vertex.getAttribute('X')), float(vertex.getAttribute('Y'))
    polygons = []
    for zz in xy:
        polygon = []
        for k in range(len(zz)):
            polygon.append((zz[k][0], zz[k][1]))
        polygons.append(polygon)
    return polygon2mask(w, h, polygons)