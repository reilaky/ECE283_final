import matplotlib.pyplot as plt
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


def plot_images(content_image, style_image, mixed_image):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Plot the content-image.
    # Note that the pixel-values are normalized to
    # the [0.0, 1.0] range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0)
    ax.set_xlabel("Content")

    # Plot the mixed-image.
    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0)
    ax.set_xlabel("Mixed")

    # Plot the style-image
    ax = axes.flat[2]
    ax.imshow(style_image / 255.0)
    ax.set_xlabel("Style")

    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

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
            layer_losses.append(loss)

        # The combined loss for all layers
        total_loss = tf.reduce_mean(layer_losses)

    return total_loss

def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
           tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))

    return loss

#-----------------------------------------------------------------------------------------------------------------------
img_size_limit = 400

content_filename = 'images/ucsb.jpg'
content_image = load_image(content_filename, max_size=img_size_limit)

style_filename = 'images/style2.jpg'
style_image = load_image(style_filename, max_size=img_size_limit)

content_layer_ids  = [4]
style_layer_ids = list(range(13))

# img = style_transfer(content_image=content_image,
#                      style_image=style_image,
#                      content_layer_ids=content_layer_ids,
#                      style_layer_ids=style_layer_ids,
#                      weight_content=1.5,
#                      weight_style=10.0,
#                      weight_denoise=0.3,
#                      num_iterations=600,
#                      step_size=10.0)

#---------------------- Sytle Transfer ----------------------
weight_content=1.5
weight_style=10.0
weight_denoise=0.3
num_iterations = 600
step_size = 10

model = vgg16.VGG16()

session = tf.InteractiveSession(graph=model.graph)

# Print the names of the content-layers.
print("Content layers:")
print(model.get_layer_names(content_layer_ids))
print()

# Print the names of the style-layers.
print("Style layers:")
print(model.get_layer_names(style_layer_ids))
print()

# Create the loss-function for the content-layers and -image.
loss_content = create_content_loss(session=session,
                                   model=model,
                                   content_image=content_image,
                                   layer_ids=content_layer_ids)

# Create the loss-function for the style-layers and -image.
loss_style = create_style_loss(session=session,
                               model=model,
                               style_image=style_image,
                               layer_ids=style_layer_ids)

# Create the loss-function for the denoising of the mixed-image.
loss_denoise = create_denoise_loss(model)

adj_content = tf.Variable(1e-10, name='adj_content')
adj_style = tf.Variable(1e-10, name='adj_style')
adj_denoise = tf.Variable(1e-10, name='adj_denoise')

# Initialize the adjustment values for the loss-functions.
session.run([adj_content.initializer,
             adj_style.initializer,
             adj_denoise.initializer])

# Create TensorFlow operations for updating the adjustment values.
# These are basically just the reciprocal values of the
# loss-functions, with a small value 1e-10 added to avoid the
# possibility of division by zero.
update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

loss_combined = weight_content * adj_content * loss_content + \
                weight_style * adj_style * loss_style + \
                weight_denoise * adj_denoise * loss_denoise

gradient = tf.gradients(loss_combined, model.input)

# List of tensors that we will run in each optimization iteration.
run_list = [gradient, update_adj_content, update_adj_style, \
            update_adj_denoise]

# The mixed-image is initialized with random noise.
# It is the same size as the content-image.
# where we first init it
mixed_image_filename = 'images/white_noise.jpg'
temp = image.load_img(mixed_image_filename, target_size=(content_image.shape[0], content_image.shape[1]))
mixed_image = image.img_to_array(temp)

for i in range(num_iterations):
    # Create a feed-dict with the mixed-image.
    feed_dict = model.create_feed_dict(image=mixed_image)

    # Calculate the value of the gradient and update the adjustment values.
    grad, adj_content_val, adj_style_val, adj_denoise_val \
        = session.run(run_list, feed_dict=feed_dict)

    grad = np.squeeze(grad)

    # Scale the step-size according to the gradient-values.
    step_size_scaled = step_size / (np.std(grad) + 1e-8)

    # Update the image by following the gradient descent
    mixed_image -= grad * step_size_scaled

    # Ensure the image has valid pixel-values between 0 and 255.
    mixed_image = np.clip(mixed_image, 0.0, 255.0)

    # Display status once every 10 iterations, and the last.
    if ((i+1) % 200 == 0):
        print()
        print("Iteration:", i)

        # Print adjustment weights for loss-functions.
        msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
        print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

        # Plot the content-, style- and mixed-images.
        plot_images(content_image=content_image, style_image=style_image, mixed_image=mixed_image)

session.close()

plt.show()
