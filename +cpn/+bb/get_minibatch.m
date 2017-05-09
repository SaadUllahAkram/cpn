function [input_blobs] = get_minibatch(conf, image_roidb)

num_images = length(image_roidb);
rois_per_image = conf.batch_size;
fg_rois_per_image = round(rois_per_image * conf.fg_fraction);

for i = 1:num_images
    im_blob = single(conf.imread(image_roidb(i).image_id)) - conf.image_means;% Get the input image blob
    [labels,label_weights,bbox_targets,bbox_loss] = sample_rois(conf, image_roidb(i), fg_rois_per_image, rois_per_image);
    
    % get fcn output size
    img_size = round(image_roidb(i).im_size * 1);
    output_size = cell2mat([conf.output_map.values({img_size(1)}), conf.output_map.values({img_size(2)})]);
    
    assert(isequal(img_size(1:2), [size(im_blob, 1) size(im_blob, 2)]));
    
    labels_blob = reshape(labels, size(conf.anchors, 1), output_size(1), output_size(2));
    label_weights_blob = reshape(label_weights, size(conf.anchors, 1), output_size(1), output_size(2));
    bbox_targets_blob = reshape(bbox_targets', size(conf.anchors, 1)*4, output_size(1), output_size(2));
    bbox_loss_blob = reshape(bbox_loss', size(conf.anchors, 1)*4, output_size(1), output_size(2));
    
    % permute from [channel, height, width], where channel is the
    % fastest dimension to [width, height, channel]
    labels_blob = permute(labels_blob, [3, 2, 1]);
    label_weights_blob = permute(label_weights_blob, [3, 2, 1]);
    bbox_targets_blob = permute(bbox_targets_blob, [3, 2, 1]);
    bbox_loss_blob = permute(bbox_loss_blob, [3, 2, 1]);
end

if size(im_blob, 3) == 3% permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
end
im_blob = single(permute(im_blob, [2, 1, 3, 4]));
labels_blob = single(labels_blob);
labels_blob(labels_blob > 0) = 1; %to binary lable (fg and bg)

label_weights_blob = single(label_weights_blob);
bbox_targets_blob = single(bbox_targets_blob);
bbox_loss_blob = single(bbox_loss_blob);

assert(~isempty(im_blob));
assert(~isempty(labels_blob));
assert(~isempty(label_weights_blob));
assert(~isempty(bbox_targets_blob));
assert(~isempty(bbox_loss_blob));

input_blobs = {im_blob, labels_blob, label_weights_blob, bbox_targets_blob, bbox_loss_blob};
end


% sample pos/neg anchors and the anchor targets
function [labels, label_weights, bbox_targets, bbox_loss_weights] = sample_rois(conf, image_roidb, fg_rois_per_image, rois_per_image)
bbox_targets = image_roidb.bbox_targets{1};
ex_asign_labels = bbox_targets(:, 1);
fg_inds = find(bbox_targets(:, 1) > 0);% Select foreground ROIs as those with >= FG_THRESH overlap
bg_inds = find(bbox_targets(:, 1) < 0);% Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)

% select foreground
fg_num = min(fg_rois_per_image, length(fg_inds));
bg_num = min(rois_per_image - fg_num, length(bg_inds));

if conf.balanced
    fg_num = min(fg_num, bg_num);
    bg_num = fg_num;
end

fg_inds = fg_inds(randperm(length(fg_inds), fg_num));
bg_inds = bg_inds(randperm(length(bg_inds), bg_num));

labels = zeros(size(bbox_targets, 1), 1);
% set foreground labels
labels(fg_inds) = ex_asign_labels(fg_inds);
assert(all(ex_asign_labels(fg_inds) > 0));

label_weights = zeros(size(bbox_targets, 1), 1);
label_weights(fg_inds) = 1;% set foreground labels weights
label_weights(bg_inds) = conf.bg_weight;% set background labels weights

bbox_targets = single(full(bbox_targets(:, 2:end)));
bbox_loss_weights = bbox_targets * 0;
bbox_loss_weights(fg_inds, :) = 1;
end
