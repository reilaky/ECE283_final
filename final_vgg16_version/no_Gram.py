import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import numpy as np
import PIL.Image
import vgg16
from keras.preprocessing import image

vgg16.maybe_download()

#-----------------------------------------------------------------------------------------------------------------------
# Functions
def load_image(filename, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        # Calculate the appropriate rescale-factor for
        # ensuring a max height and width, while keeping
        # the proportion between them.
        factor = max_size / np.max(image.size)

        # Scale the image's height and width.
        size = np.array(image.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        image = image.resize(size, PIL.Image.LANCZOS)

    # Convert to numpy floating-point array.
    return np.float32(image)

def gram_matrix(tensor):
    # gram matrix of the feature activations of a style layer

    shape = tensor.get_shape()

    # Get the number of feature channels for the input tensor,
    num_channels = int(shape[3])

    # flattens the contents of each feature-channel.
    matrix = tf.reshape(tensor, shape=[-1, num_channels])

    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram

def create_style_loss(session, model, style_image, layer_ids):
    feed_dict = model.create_feed_dict(image=style_image)

    layers = model.get_layer_tensors(layer_ids)

    with model.graph.as_default():
        gram_layers = [gram_matrix(layer) for layer in layers]

        # Calculate Gram-matrices
        values = session.run(gram_layers, feed_dict=feed_dict)

        layer_losses = []

        for value, gram_layer in zip(values, gram_layers):
            loss = tf.reduce_mean(tf.square(gram_layer - tf.constant(value)))

            # loss += 0.00002 * tf.reduce_mean(tf.nn.l2_loss(gram_layer))
            # loss += 100*tf.reduce_sum(tf.losses.absolute_difference(gram_layer, tf.zeros_like(gram_layer)))

            layer_losses.append(loss)

        # The combined loss for all layers
        total_loss = tf.reduce_mean(layer_losses)

    return total_loss

#-----------------------------------------------------------------------------------------------------------------------
# Prepare images
content_size_limit = 500
style_size_limit = 500

content_filename = 'images/goleta.jpg'
# content_filename = 'images/ucsb.jpg'
content_image = load_image(content_filename, max_size=content_size_limit)

# style_filename = 'images/star.jpg'
style_filename = 'images/style3.jpg'
# style_filename = 'images/style9.jpg'
# style_filename = 'images/Pablo Picasso - Seated Woman, 1946.jpg'
# style_filename = 'images/theScream.jpg'
# style_filename = 'images/style5.jpg'
# style_filename = 'images/'
# style_filename = 'images/'

style_image = load_image(style_filename, max_size=style_size_limit)

# style_layer_ids = [0, 2, 4, 7, 10]
style_layer_ids = list(range(13))
# style_layer_ids = list(range(7,13))
# style_layer_ids = list(range(2))
# style_layer_ids = [0, 1, 10, 11, 12]
# style_layer_ids = [0, 1, 7, 8, 9]
# style_layer_ids = [0, 1, 4, 5, 6]
# style_layer_ids = list(range(10))
# style_layer_ids = list(range(7))
# style_layer_ids = list(range(4))
# style_layer_ids = list(range(2,4))
# style_layer_ids = list(range(4,7))
# style_layer_ids = list(range(7,10))
# style_layer_ids = list(range(10,13))

#-----------------------------------------------------------------------------------------------------------------------
# Sytle Transfer
weight_style = 1.0
step_size = 10
# num_iterations = 200
num_iterations = 50
disp_interval = num_iterations

model = vgg16.VGG16()

session = tf.InteractiveSession(graph=model.graph)

# Print the names of the style-layers.
print("Style layers:")
print(model.get_layer_names(style_layer_ids))
print()

# Create the loss-function for the style-layers and -image.
loss_style = create_style_loss(session=session,
                               model=model,
                               style_image=style_image,
                               layer_ids=style_layer_ids)

adj_content = tf.Variable(1e-10, name='adj_content')
adj_style = tf.Variable(1e-10, name='adj_style')

# Initialize the adjustment values for the loss-functions.
session.run([adj_content.initializer, adj_style.initializer])

# Create TensorFlow operations for updating the adjustment values.
# These are basically just the reciprocal values of the loss-functions
update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))

loss_combined = weight_style * adj_style * loss_style

gradient = tf.gradients(loss_combined, model.input)

# List of tensors that we will run in each optimization iteration.
run_list = [gradient, update_adj_style]

# The mixed-image is initialized with random noise with the same size as the content-image.
# mixed_image_filename = 'images/ucsb.jpg'
# mixed_image_filename = 'images/goleta.jpg'

temp = image.load_img(content_filename, target_size=(content_image.shape[0], content_image.shape[1]))
mixed_image = image.img_to_array(temp)

row_num = int(num_iterations/disp_interval)
plt.figure()
plt.get_current_fig_manager().window.wm_geometry("1400x900+20+20")
gs = gridspec.GridSpec(row_num, 3)
gs.update(wspace=0.05, hspace=0.2)
plt_count = 0

for i in range(num_iterations):
    # Create a feed-dict with the mixed-image.
    feed_dict = model.create_feed_dict(image=mixed_image)

    # Calculate the value of the gradient and update the adjustment values.
    grad, adj_style_val = session.run(run_list, feed_dict=feed_dict)

    grad = np.squeeze(grad)

    # Scale the step-size according to the gradient-values.
    step_size_scaled = step_size / (np.std(grad) + 1e-8)

    # Update the image by following the gradient descent
    mixed_image -= grad * step_size_scaled

    # Ensure the image has valid pixel-values between 0 and 255.
    mixed_image = np.clip(mixed_image, 0.0, 255.0)

    if ((i+1) % disp_interval == 0):
        msg = "Iteration {0:d}: Weight Adj for Style: {1:.2e}"
        print(msg.format(i+1, adj_style_val))

        plt.subplot(gs[plt_count])
        plt.imshow(content_image / 255.0)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("Iteration {:d}".format(i+1))
        plt.title("Content")

        plt.subplot(gs[plt_count+1])
        plt.imshow(mixed_image / 255.0)
        plt.axis('off')
        plt.title("Mixed")

        plt.subplot(gs[plt_count+2])
        plt.imshow(style_image / 255.0)
        plt.axis('off')
        plt.title("Style")

        plt_count += 3

session.close()

plt.show()

