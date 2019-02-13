import aircv as ac
from PIL.ImageDraw import Draw
import matplotlib.pyplot as plt
from keras.preprocessing import image

base_dir='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/'
TRAIN = base_dir+'train_backup/'
TEST = base_dir+'test_backup/'
P2SIZE = base_dir+'metadata/p2size.pickle'
BB_DF = base_dir+'metadata/bounding_boxes.csv'

def find_image_pos(img1_path, img2_path):
    img1=ac.imread(img1_path)
    img2=ac.imread(img2_path)
     
    pos=ac.find_template(img1,img2)
    print (pos)
    print (pos.get('rectangle'))
    x0=pos.get('rectangle')[0][0]
    y0=pos.get('rectangle')[0][1]
    x1=pos.get('rectangle')[3][0]
    y1=pos.get('rectangle')[3][1]
     
    return x0,y0,x1,y1
     
def test():               
    img1_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/train_backup/0ef8ede21.jpg'
    img2_path='/Users/chendanxia/sophie/kaggle/humpback-whale-identification/data/clean_train/0ef8ede21.jpg'
    x0,y0,x1,y1=find_image_pos(img1_path, img2_path)
    
    img1_1=image.load_img(img1_path)
    img2=image.load_img(img2_path)
    print img1_1.size
    print img2.size
    draw=Draw(img1_1)
    draw.rectangle([x0,y0,x1,y1], outline='red')
    plt.imshow(img1_1)
    plt.show()

test()