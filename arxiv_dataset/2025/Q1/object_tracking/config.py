import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Stop spam from tensorflow
import tensorflow as tf

sessions = [28]
trials = [1]
is_retro = [False]

all_camera_names = ['table-side', 'table-top', 'back', 'counter-top', 'ceiling', 'front']
rect_by_cam = ['table', 'table', 'table', 'counter', 'table', 'table'] # the rectangle to perform optimization of camera params on.

xdof = 6 # degrees of freedom per camera

alpha = 750 # focal length in pixels

corners = ['ul', 'ur', 'lr', 'll'] # how the corners of the table/counter are called

groundtruth_w = { # corner positions in world coords: upper left, upper right, lower right, lower left
    'table': tf.constant([[0.38, -1.6, 0.73], [0.38, 0, 0.73], [1.18, 0, 0.73], [1.18, -1.6, 0.73]], dtype=tf.float64),
    'counter': tf.constant([[-1.66, -0.415, 0.74], [-1.66, 1.385, 0.74], [-1.06, 1.385, 0.74], [-1.06, -0.415, 0.74]], dtype=tf.float64),
    'origin': tf.constant([[0, 0, 0]], dtype=tf.float64)
}

# does only one instance of this tableware_class exist?
predifined_classes = {'bowl--salad': True, 'bowl--cooker': True, 'plate--pasta': True, 'bread': True, 'butter': True, 'jam': True, 'nutella': True, 'salt': True, 'shaker--pepper': True, 'sugar': True, 'cereal': True, 'milk': True, 'coffee': True ,'wine--bottle': True, 'water': True, 'bowl--cereal': False, 'plate': False, 'knife-bread': True, 'utensil, pasta': True, 'ladle': True, 'utensil-spoon, salad': True, 'utensil-fork, salad': True, 'teaspoon': False, 'tablespoon': False, 'fork': False, 'knife': False, 'glass-water': False, 'glass--wine': False, 'cup-coffee': False}

img_shape = (720, 1280, 3)

example_annotation = {'table-side': {'counter': {'ul': [721, 168],
   'ur': None,
   'lr': [1124, 225],
   'll': [728, 211]},
  'table': {'ul': [389, 392],
   'ur': [928, 401],
   'lr': [1044, 619],
   'll': [284, 610]},
  'origin': {'ul': None}},
 'table-top': {'counter': {'ul': None, 'ur': None, 'lr': None, 'll': None},
  'table': {'ul': [994, 519],
   'ur': [324, 532],
   'lr': [330, 198],
   'll': [980, 195]},
  'origin': {'ul': [418, 611]}},
 'back': {'counter': {'ul': None,
   'ur': [1160, 570],
   'lr': [964, 593],
   'll': [795, 216]},
  'table': {'ul': [464, 110],
   'ur': [452, 309],
   'lr': [229, 324],
   'll': [304, 116]},
  'origin': {'ul': [559, 420]}},
 'counter-top': {'counter': {'ul': [971, 496],
   'ur': [303, 466],
   'lr': [307, 243],
   'll': [985, 270]},
  'table': {'ul': None, 'ur': None, 'lr': None, 'll': None},
  'origin': {'ul': None}},
 'ceiling': {'counter': {'ul': [740, 697],
   'ur': [417, 683],
   'lr': [416, 577],
   'll': [741, 585]},
  'table': {'ul': [951, 308],
   'ur': [651, 315],
   'lr': [647, 171],
   'll': [949, 159]},
  'origin': {'ul': [668, 376]}},
 'front': {'counter': {'ul': [64, 297],
   'ur': None,
   'lr': None,
   'll': [236, 310]},
  'table': {'ul': [690, 675],
   'ur': [696, 261],
   'lr': [918, 270],
   'll': [1092, 691]},
  'origin': {'ul': [600, 409]}}}