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

def create_content_loss(session, model, content_image, layer_ids):
    feed_dict = model.create_feed_dict(image=content_image)

    layers = model.get_layer_tensors(layer_ids)

    # Calculate the output values of those layers when feeding the content-image to the model.
    values = session.run(layers, feed_dict=feed_dict)

    with model.graph.as_default():
        layer_losses = []

        for value, layer in zip(values, layers):
            # The loss-function for this layer: Mean Squared Error
            loss =  tf.reduce_mean(tf.square(layer - tf.constant(value)))

            layer_losses.append(loss)

        # The combined loss for all layers
        total_loss = tf.reduce_mean(layer_losses)

    return total_loss

def gram_matrix(tensor):
    # gram matrix of the feature activations of a style layer

    shape = tensor.get_shape()

    # Get the number of feature channels for the input tensor,
    num_channels = int(shape[3])

    # flattens the contents of each feature-channel.
    matrix = tf.reshape(tensor, shape=[-1, num_channels])

    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram

#-----------------------------------------------------------------------------------------------------------------------
# Prepare images
content_size_limit = 500
style_size_limit = 500

content_filename = 'images/goleta.jpg'
# content_filename = 'images/white_noise.jpg'
content_image = load_image(content_filename, max_size=content_size_limit)

# content_layer_ids  = [4]
# content_layer_ids  = [8]
content_layer_ids  = [11]

#-----------------------------------------------------------------------------------------------------------------------
# Sytle Transfer
weight_content = 1.0
step_size = 10
num_iterations = 200
disp_interval = 100

model = vgg16.VGG16()

session = tf.InteractiveSession(graph=model.graph)

# Print the names of the content-layers.
print("Content layers:")
print(model.get_layer_names(content_layer_ids))
print()

# Create the loss-function for the content-layers and -image.
loss_content = create_content_loss(session=session,
                                   model=model,
                                   content_image=content_image,
                                   layer_ids=content_layer_ids)

adj_content = tf.Variable(1e-10, name='adj_content')
adj_style = tf.Variable(1e-10, name='adj_style')

# Initialize the adjustment values for the loss-functions.
session.run([adj_content.initializer, adj_style.initializer])

# Create TensorFlow operations for updating the adjustment values.
# These are basically just the reciprocal values of the loss-functions
update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))

loss_combined = weight_content * adj_content * loss_content

gradient = tf.gradients(loss_combined, model.input)

# List of tensors that we will run in each optimization iteration.
run_list = [gradient, update_adj_content]

# The mixed-image is initialized with random noise with the same size as the content-image.
# mixed_image_filename = 'images/ucsb.jpg'
# mixed_image_filename = 'images/goleta.jpg'
mixed_image_filename = 'images/white_noise.jpg'

temp = image.load_img(mixed_image_filename, target_size=(content_image.shape[0], content_image.shape[1]))
mixed_image = image.img_to_array(temp)

row_num = int(num_iterations/disp_interval)
plt.figure()
plt.get_current_fig_manager().window.wm_geometry("1400x900+20+20")
gs = gridspec.GridSpec(row_num, 2)
gs.update(wspace=0.05, hspace=0.2)
plt_count = 0

for i in range(num_iterations):
    # Create a feed-dict with the mixed-image.
    feed_dict = model.create_feed_dict(image=mixed_image)

    # Calculate the value of the gradient and update the adjustment values.
    grad, adj_content_val = session.run(run_list, feed_dict=feed_dict)

    grad = np.squeeze(grad)

    # Scale the step-size according to the gradient-values.
    step_size_scaled = step_size / (np.std(grad) + 1e-8)

    # Update the image by following the gradient descent
    mixed_image -= grad * step_size_scaled

    # Ensure the image has valid pixel-values between 0 and 255.
    mixed_image = np.clip(mixed_image, 0.0, 255.0)

    if ((i+1) % disp_interval == 0):
        msg = "Iteration {0:d}: Weight Adj. for Content: {1:.2e}"
        print(msg.format(i+1,adj_content_val))

        plt.subplot(gs[plt_count])
        plt.imshow(content_image / 255.0)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("Iteration {:d}".format(i))
        plt.title("Content")

        plt.subplot(gs[plt_count+1])
        plt.imshow(mixed_image / 255.0)
        plt.axis('off')
        plt.title("Filtered")

        plt_count += 2

session.close()

plt.show()

