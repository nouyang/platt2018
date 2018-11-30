import math


from PIL import Image
from PIL import ImageDraw
import math

L=512; W=512
image = Image.new("1", (L, W))
draw = ImageDraw.Draw(image)



vertices = makeRectangle(50, 60, 20*math.pi/180, offset=(L/2+20, -20+W/2))
draw.polygon(vertices, fill=1)

image.save("test.png")
