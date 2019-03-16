############################################################
#  Region Proposal Network (RPN)
############################################################
 
#For 3D image, we are going to call depth(D) as dimension and depth_network as depth of feature network. Remember!
 
def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.
    feature_map: backbone features [batch, height, width, depth, depth_network]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    Returns:
        Total number of anchors will also change in 3D images. Anchors_per_location is for every pixel and number of pixels will change and will become H*W*D.
        rpn_class_logits: [batch, H * W * D * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * D * anchors_per_location, 2] Anchor classifier probabilities.
        Note: To avoid confusion with d in dx etc. I have denoted the delta of depth as d(dep). For 3D images, bounding boxes will also be 3D, it's known.
        rpn_bbox: [batch, H * W * D * anchors_per_location, (dy, dx, dz, log(dh), log(dw), log(d(dep)] Deltas to be
                  applied to anchors.
    """
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    shared = KL.Conv3D(512, (3, 3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)
 
    # Anchor Score. [batch, height, width, depth, anchors per location * 2].
    x = KL.Conv3D(2 * anchors_per_location, (1, 1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)
    #We can change these number of filters to also num_of_classes*anchors_per_location. We are not changing this here since we are assuming classes in object can belong too is still same
    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)
 
    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)
 
    # Bounding box refinement. [batch, H, W, D, anchors per location * depth]
    # where depth is [x, y, z, log(w), log(h), log(dep)]
    #Since our bounding box will mention 6 attributes as per 3D image (hence it's depth is 6). This depth is different from the other one!
    x = KL.Conv3D(anchors_per_location * 6, (1, 1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)
 
    # Reshape to [batch, anchors, 6]
    #Reshapes to batch, H*W*D*Anchors_per_location, 6
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 6]))(x)
 
    return [rpn_class_logits, rpn_probs, rpn_bbox]
