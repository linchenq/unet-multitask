import os
import json
import numpy as np
import hdf5storage

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imsave
from sklearn.model_selection import train_test_split

from lxml import etree

from tqdm import tqdm

# image size
IMAGE_SIZE = 512.0

# path of the standard architecture
MESH_PATH = "C:/Research/LumbarSpine/Data/json/GeneralSpineMesh.json"

# path of origin mats, generated jpeg/xml path
ROOT_PATH = "C:/Research/LumbarSpine/Data/SpineMeshTest/"
JPEG_PATH = "./images/"
XML_PATH = "./annotations/"

# path of training/validing sets
LABEL_TXT = "./labels/"

# txt files relevant to yolov3
TRAIN_TXT = "./train.txt"
VALID_TXT = "./valid.txt"
TEST_TXT = "./test.txt"

TEST_TARGET_PATH ="./samples/"

# TAGS
TAGS = ['D1', 'D2', 'D3', 'D4', 'D5', 'S']

# DEBUG MODE
DEBUG = False

class Unit(object):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

        # load general spinemesh
        with open(MESH_PATH, 'r') as f:
            self.point_dict = json.load(f)

        # initial raw_image and 0-indexed points
        self._init_bbox()

    def _init_bbox(self):
        mat_data = hdf5storage.loadmat(self.filepath)
        self.raw_image, self.raw_points = mat_data['ImageShapePair'][:, 0][0], mat_data['ImageShapePair'][:, 1][0]

        # the points in .mat are 1-indexed. so convert to 0-indexed.
        self.ant_points = np.transpose(self.raw_points[0:2])
        self.ant_points[:, 0], self.ant_points[:, 1] = self.ant_points[:, 0] - 1., self.ant_points[:, 1] -1.

        assert self.ant_points.shape == (1996, 2)

    def gen_jpeg(self, basename, path=JPEG_PATH):
        path = path + basename + ".jpg"
        imsave(path, self.raw_image, cmap = plt.get_cmap('gray'))

    def gen_xml(self, basename, path=XML_PATH):
        path = path + basename + ".xml"
        xml_dict = {}

        if DEBUG:
            plt.figure()

        for tag in TAGS:
            target = self.ant_points[self.point_dict[tag], :]
            target = target.astype(np.float32)

            x, y, w, h = cv2.boundingRect(target)

            if DEBUG:
                plt.gca().add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'))

            xmin, ymin, xmax, ymax = x, y, x+w, y+h
            xml_dict[tag] = (xmin, ymin, xmax, ymax)

        if DEBUG:
            plt.imshow(self.raw_image, cmap='gray')

        save_xml = XmlUnit(basename, (512, 512, 1), xml_dict)
        save_xml.output(path=path)

    def gen_txt(self, basename, path=LABEL_TXT):
        path = path + basename + ".txt"
        txt_dict = {}

        assert IMAGE_SIZE == 512.0

        for i in range(len(TAGS)):
            target = self.ant_points[self.point_dict[TAGS[i]], :]
            target = target.astype(np.float32)

            x, y, w, h = cv2.boundingRect(target)
            x, y, w, h = (x + w/2.0), (y + h/2.0), w, h
            x, y, w, h = x/IMAGE_SIZE, y/IMAGE_SIZE, w/IMAGE_SIZE, h/IMAGE_SIZE

            txt_dict[i] = (x, y, w, h)

        with open(path, 'w') as f:
            for k, v in txt_dict.items():
                print('{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f}'.format(k, v[0], v[1], v[2], v[3]), file=f)


class XmlUnit(object):
    
    FOLDER = 'GT30'
    DATABASE = 'The Lumbar GT30 Database'
    ANNOTATION = 'LUMBAR GT30'

    def __init__(self, filename, size, o_dict):
        '''
        Parameters Defination:
            filename : str, the name of the image with suffix
            size: (height, width, depth/channel), tuple
            o_dict : { name : (xmin, ymin, xmax, ymax) }

        Important notes:
            xmin, ..., ymax SHOULD be 1-based pixels
        '''
        super().__init__()
        self.filename = filename
        self.height, self.width, self.depth = size
        self.o_dict = o_dict

        self.root = self._generate()

    def output(self, path=None):
        s = etree.tostring(self.root, pretty_print=True)
        if path is not None:
            with open(path, 'wb') as f:
                f.write(s)
        else:
            print(s)

    def _generate(self):
        # create xml
        root = etree.Element('annotation')
        root.append(self._generate_child('folder', XmlUnit.FOLDER))
        root.append(self._generate_child('filename', self.filename))

        # source
        source = etree.Element('source')
        root.append(source)
        source.append(self._generate_child('database', XmlUnit.DATABASE))
        source.append(self._generate_child('annotation', XmlUnit.ANNOTATION))

        # size
        t_size = etree.Element('size')
        root.append(t_size)
        t_size.append(self._generate_child('width', str(self.width)))
        t_size.append(self._generate_child('height', str(self.height)))
        t_size.append(self._generate_child('depth', str(self.depth)))

        # object
        t_object = etree.Element('object')
        root.append(t_object)
        for k, v in self.o_dict.items():
            t_object.append(self._generate_child('name', k))

            #bnd box
            bndbox = etree.Element('bndbox')
            t_object.append(bndbox)

            bndbox.append(self._generate_child('xmin', str(v[0])))
            bndbox.append(self._generate_child('ymin', str(v[1])))
            bndbox.append(self._generate_child('xmax', str(v[2])))
            bndbox.append(self._generate_child('ymax', str(v[3])))

        return root

    def _generate_child(self, prefix, context=None):
        child = etree.Element(prefix)
        if context is not None:
            child.text = context
        return child

# dataset generation

def gen_txt(txt_path, filenames):
    print ("Generating dataset relevant txts...")
    with open(txt_path, 'w') as out:
        for filename in tqdm(filenames):
            suffix = os.path.splitext(filename)[0]
            print("C:/Research/LumbarSpine/Github/unet-multitask/datasets/localization/images/{}.jpg".format(suffix), file=out)

def dataset_split(root_folder, train_txt, valid_txt, test_txt):
    root = os.listdir(root_folder)

    train, test, _, _ = train_test_split(root,
                                         range(0, len(root)),
                                         test_size=0.1,
                                         random_state=17)

    train, valid, _, _ = train_test_split(train,
                                          range(0, len(train)),
                                          test_size=0.1,
                                          random_state=17)

    gen_txt(TRAIN_TXT, train)
    gen_txt(VALID_TXT, valid)
    gen_txt(TEST_TXT, test)

    gen_testset(test, "")

def gen_testset(test_list, target):
    print ("Generating datasets...")
    for file in tqdm(test_list):
        filename = os.path.splitext(file)
        filepath = os.path.join(ROOT_PATH, file)

        if not os.path.isdir(filepath) and filename[-1] == ".mat":
            tmp = Unit(filepath)
            tmp.gen_jpeg(basename=filename[0], path=TEST_TARGET_PATH)


if __name__ == '__main__':

    # TODO : Reminder as below
    # Remember to modify ROOT_PATH to the final folders

    files = os.listdir(ROOT_PATH)
    print ("Generating annotations/images...")
    for file in tqdm(files):
        filename = os.path.splitext(file)
        filepath = os.path.join(ROOT_PATH, file)

        if not os.path.isdir(filepath) and filename[-1] == ".mat":
            tmp = Unit(filepath)

            tmp.gen_jpeg(basename=filename[0])

            tmp.gen_xml(basename=filename[0])

            tmp.gen_txt(basename=filename[0])

    dataset_split(ROOT_PATH, TRAIN_TXT, VALID_TXT, TEST_TXT)


