import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lib.utils.demo_utils import download_crosswalk


images = [
    "./data/test_images/cw_example_1.jpg",
    "./data/test_images/cw_example_2.jpg"
]

download_crosswalk()
interpreter = tf.lite.Interpreter(model_path="./data/segm_data/uc?id=1ZxObpKaG8bLLwvwBY0cmXj0ptHm-z54C&export=download")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for idx in range(len(images)):
    input_shape = input_details[0]['shape']

    img = cv2.imread(images[idx])
    img = cv2.resize(img, (416, 416))
    input_data = np.array(np.expand_dims(img, 0) / 255, dtype=np.float32)
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_rect = interpreter.get_tensor(output_details[0]['index'])
    output_eval = interpreter.get_tensor(output_details[1]['index'])

    plt.close('all')
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for num, data in enumerate(output_eval[0]):
        person, crosswalk = data
        if person > 0.7 or crosswalk > 0.7:
            xpos, ypos, w, h = output_rect[0,num,:]   
            x = max(0, xpos-w/2)
            y = max(0, ypos-h/2)
            if person > crosswalk : 
                rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
            else : 
                rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='b',facecolor='none')
                ax.add_patch(rect)
    plt.show()
