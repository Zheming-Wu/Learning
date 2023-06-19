from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
img_path = "data/train/ants_image/470127037_513711fd21.jpg"
img = Image.open(img_path)
img_array = np.array(img)

print(img_array.shape)

writer.add_image('test', img_array, 2, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y = x^4", i*i*i*i, i)
    pass

writer.close()

# tensorboard --logdir=logs --port=6006

