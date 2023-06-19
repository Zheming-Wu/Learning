from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2

img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(img)
# print(tensor_img)

cv_img = cv2.imread(img_path)
print(type(cv_img), cv_img.shape)

# python中带__方法名__的方法都是默认的运行函数时自动运行的，有什么好讨论的了

writer = SummaryWriter("logs")
writer.add_image('Tensor_img', tensor_img)
writer.close()

