import math

def makeRectangle(l, w, theta, offset=(0,0)):
    c, s = math.cos(theta), math.sin(theta)
    rectCoords = [(l/2.0, w/2.0), (l/2.0, -w/2.0), (-l/2.0, -w/2.0), (-l/2.0, w/2.0)]
    return [(c*x-s*y+offset[0], s*x+c*y+offset[1]) for (x,y) in rectCoords]

from PIL import Image
from PIL import ImageDraw
import math

L=512; W=512
image = Image.new("1", (L, W))
draw = ImageDraw.Draw(image)



vertices = makeRectangle(50, 60, 20*math.pi/180, offset=(L/2+20, -20+W/2))
draw.polygon(vertices, fill=1)

image.save("test.png")
