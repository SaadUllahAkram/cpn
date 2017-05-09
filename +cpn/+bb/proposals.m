function [bb_nms, bb_all, anchors_all] = proposals(conf, cpn_net, ims)
% takes as input a net and and image/or a cell of images and returns proposals (after opts_nms)
%
% Outputs:
%     bb_nms: Nx1 cell array of bboxes for each image. bboxes are in 'b'-format [top_left_corner, bbox_size]
%
if length(ims) >= 10 && iscell(ims)
    verbose = 1;
else
    verbose = 0;
end
db = conf.debug;

conf = update_struct_loc(conf, {'feat_stride','test_stride'},{-1,0});
feat_stride = conf.feat_stride;% mdl stride
test_stride = conf.test_stride;% 0(evaluate the model once), else (shift image and obtain new bboxes)
opts_nms = conf.nms;

if ~isfield(opts_nms,'nms_score')
    opts_nms.nms_score = 0;
end

if isnumeric(ims)%todo: remove it
    im2{1} = ims;
    ims = im2;
    clear im2
end

N      = length(ims);
bb_nms = cell(N,1);
bb_all = cell(N,1);
anchors_all = cell(N,1);
if verbose
    tic_start = tic;
    fprintf('%d:  ', N)
end

for i=1:N
    szs(i,:) = size(ims{i});
end
szs = unique(szs,'rows');
if size(szs,1) > 1; init = 1;% 1: get anchors for each image
else; init = 2;% 2: get anchors for 1st image only
end

for i=1:N
    if verbose
        fprintf(repmat('\b',1, sum([ (i-1) >0 (i-1)>9 (i-1)>99])))
        fprintf('%d', i)
    end
    [boxes, scores, anchors] = proposal_im_detect_strided(test_stride, feat_stride, conf, cpn_net, ims{i}, init>0);
    if init == 2; init = 0;  end
    
    bb_all{i} = [boxes, scores];
    anchors_all{i} = anchors;
    bb_corners = cpn.utils.boxes_filter([boxes, scores],opts_nms.per_nms_topN,opts_nms.nms_overlap_thres,opts_nms.after_nms_topN,opts_nms.use_gpu);% [xmin, ymin, xmax, ymax]
    % for i=1:max(anchors_id);fprintf('%2d : %4d\n', i, sum(anchors_id==i));end
    
    if opts_nms.nms_score > 0
        bb_corners(bb_corners(:,5)<opts_nms.nms_score, :) = [];
    end
    
    bb_nms{i}  = bb_corners;
    if 1
        bb_nms{i}(:,1:4) = bia.convert.bb(bb_nms{i}(:,1:4), 'c2b');
    else% reduce PhC-Oxford bbox, leads to improvement in AP of TRA measure
        bb_corners = bia.convert.bb(bb_nms{i}(:,1:4), 'c2b');
        bb_corners(:,1:2) = bb_corners(:,1:2)+5;
        bb_corners(:,3:4) = max(1, bb_corners(:,3:4)-10);
        bb_nms{i}(:,1:4) = bb_corners;
    end
    if db
        imshow(ims{i},[]);
        bia.plot.bb([],bb_nms{i}(bb_nms{i}(:,5)>0.98,1:4))
        drawnow
    end
end
if verbose
    % count_chars = N + sum([ (1:N) >0 (1:N)>9 (1:N)>99]);% counts # of chars printed in loop above
    count_chars = sum([ N>0 N>9 N>99]);
    fprintf(repmat('\b',1,count_chars))
    fprintf('Time (per image): %1.1f sec\n', toc(tic_start)/N)
end
end


function [boxes, scores, anchors] = proposal_im_detect_strided(test_stride, feat_stride, conf, cpn_net, im, init)
if test_stride == 0
    [boxes, scores, anchors] = cpn_im_detect(conf, cpn_net, im, init>0);
elseif test_stride > 0% reduce stride between anchors
    for pad = 0:test_stride:feat_stride-1
        [boxes_loc,scores_loc,anchors_loc] = cpn_im_detect(conf, cpn_net, padarray(im,[pad pad],'symmetric','pre'), 1);
        boxes   = [boxes; boxes_loc-pad];
        scores  = [scores; scores_loc];
        anchors = [anchors; anchors_loc];
    end
end
end


function conf = update_struct_loc(conf, fields, vals)
for i=1:length(fields)
    if ~isfield(conf, fields{i})
        conf.(fields{i}) = vals{i};
    end
end
end


function [pred_boxes, scores, anchors] = cpn_im_detect(conf, caffe_net, im, init)
% pred_boxes: 1-based coordinates, [topleft bottomright corners]

conf_default = struct('test_drop_boxes_runoff_image', false, 'image_means', 128, 'cpn_resize_im', 0);
conf         = bia.utils.updatefields(conf_default, conf);
persistent anchors_ channels
%     persistent anchors_id
if nargin < 4
    init = 1;
end

sz = size(im);
if init == 1
    channels = size(caffe_net.blobs('data').get_data(), 3);
end
im_blob = single(im) - conf.image_means;

% permute data into caffe c++ memory, thus [num, channels, height, width]
if channels == 3
    if size(im, 3) == 1
        im_blob = repmat(im_blob, [1 1 3]);
    end
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
end
im_blob = permute(im_blob, [2, 1, 3, 4]);
im_blob = single(im_blob);

net_inputs = {im_blob};

% Reshape net's input blobs
caffe_net.reshape_as_input(net_inputs);
output_blobs = caffe_net.forward(net_inputs);
scores = output_blobs{2}(:, :, 2);
% Apply bounding-box regression deltas
box_deltas = output_blobs{1};
featuremap_size = [size(box_deltas, 2), size(box_deltas, 1)];
% permute from [width, height, channel] to [channel, height, width], where channel is the
% fastest dimension
box_deltas = permute(box_deltas, [3, 2, 1]);
box_deltas = reshape(box_deltas, 4, [])';
if init == 1
    anchors_ = cpn.bb.get_anchors(conf, sz, [], featuremap_size);
end
anchors = anchors_;
%     if strcmp(conf.cpn.bbox_transform, 'log')
pred_boxes = cpn.utils.fast_rcnn_bbox_transform_inv(anchors, box_deltas);
%     elseif strcmp(conf.cpn.bbox_transform, 'log2')
%         pred_boxes = fast_rcnn_bbox_transform_inv_pow2(anchors, box_deltas);
%     end

pred_boxes = clip_boxes(pred_boxes, sz(2), sz(1));
scores = reshape(scores, size(output_blobs{1}, 1), size(output_blobs{1}, 2), []);
% permute from [width, height, channel] to [channel, height, width], where channel is the
% fastest dimension
scores = permute(scores, [3, 2, 1]);
scores = scores(:);

if conf.test_drop_boxes_runoff_image
    contained_in_image = is_contain_in_image(anchors, round(size(im)));
    pred_boxes = pred_boxes(contained_in_image, :);
    scores = scores(contained_in_image, :);
    anchors = anchors(contained_in_image,1);
end

% drop too small boxes
[pred_boxes, scores, valid_ind] = filter_boxes(conf.test_min_box_size, pred_boxes, scores);
anchors = anchors(valid_ind,1);
% sort
[scores, scores_ind] = sort(scores, 'descend');
pred_boxes = pred_boxes(scores_ind, :);
anchors = anchors(scores_ind,1);
end


function [boxes, scores, valid_ind] = filter_boxes(min_box_size, boxes, scores)
widths = boxes(:, 3) - boxes(:, 1) + 1;
heights = boxes(:, 4) - boxes(:, 2) + 1;

valid_ind = widths >= min_box_size & heights >= min_box_size;
boxes = boxes(valid_ind, :);
scores = scores(valid_ind, :);
end

function boxes = clip_boxes(boxes, im_width, im_height)
boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);% x1 >= 1 & <= im_width
boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);% y1 >= 1 & <= im_height
boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);% x2 >= 1 & <= im_width
boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);% y2 >= 1 & <= im_height
end

function contained = is_contain_in_image(boxes, im_size)
contained = boxes >= 1 & bsxfun(@le, boxes, [im_size(2), im_size(1), im_size(2), im_size(1)]);
contained = all(contained, 2);
end
