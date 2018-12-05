import math

import os

from PIL import Image
from PIL import ImageDraw
import math

os.environ["SCIPY_PIL_IMAGE_VIEWER"] = "/usr/bin/eog"

L=512; W=512
image = Image.new("1", (L, W))
draw = ImageDraw.Draw(image)


# Calc rectangle vertices. makeRectangle() credit Sparkler, stackoverflow, feb 17
# offset is to center of rectangle
def makeRectangle(l, w, theta, offset=(0, 0)):
    c, s = math.cos(theta), math.sin(theta)
    rectCoords = [(l/2.0, w/2.0), (l/2.0, -w/2.0), (-l/2.0, -w/2.0), (-l/2.0, w/2.0)]
    return [(c*x-s*y+offset[0], s*x+c*y+offset[1]) for (x, y) in rectCoords]


# vertices = makeRectangle(50, 60, 20*math.pi/180, offset=(L/2+20, -20+W/2))
vertices = makeRectangle(50, 50,0, offset=(100,100))
vertices2 = makeRectangle(50, 50,0, offset=(0,0))
draw.polygon(vertices, fill=1)
draw.polygon(vertices2, fill=1)

image.show()

image.save("test.png")
