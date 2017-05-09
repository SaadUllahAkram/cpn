function train(conf, data_train, data_val, data_debug)

cpn.io.imread()
cpn.io.gtread()

train_dir = conf.paths.dir;

path_final = fullfile(train_dir,  sprintf('final%s', conf.paths.id));
path_config = fullfile(train_dir, sprintf('config%s.mat', conf.paths.id));
path_solver = fullfile(train_dir, sprintf('solver%s.prototxt', conf.paths.id));
path_solver_final = fullfile(train_dir, sprintf('final_solver%s.prototxt', conf.paths.id));
path_def_test = fullfile(train_dir, sprintf('test%s.prototxt', conf.paths.id));
path_def_test_final = fullfile(train_dir, sprintf('final_test%s.prototxt', conf.paths.id));
path_def_train = fullfile(train_dir, sprintf('train%s.prototxt', conf.paths.id));
path_def_train_final = fullfile(train_dir, sprintf('final_train%s.prototxt', conf.paths.id));
path_init_file = conf.paths.init_net_file;

if exist(path_final, 'file');  return;   end

conf_seg = conf;%#ok<NASGU>
save(path_config, 'conf_seg')

roidb_train = data_train.roidb;

timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
caffe.init_log(fullfile(train_dir, 'caffe_'));% adds timestamp in caffe
diary(fullfile(train_dir, sprintf('matlab_%s.txt', timestamp)));

iter_prev_run = 0;
if conf.resume_train
    [path_init_file, iter_prev_run, path_solver] = cpn.utils.resume_training(train_dir, path_solver);
end

bia.print.fprintf('Red', 'Training CPN-SEG\n')
t_start = tic;
caffe_solver = caffe.Solver(path_solver);
if ~isempty(path_init_file) && exist(path_init_file,'file')
    bia.print.fprintf('red',sprintf('\nResuming training from: %s\n', path_init_file))
    caffe_solver.net.copy_from(path_init_file);
end
if isempty(data_val.roidb); conf.do_val = 0;    end
check_gpu_memory(conf, caffe_solver, data_train, data_val, conf.do_val);

bia.caffe.print_sz(caffe_solver.net);
bia.print.struct(conf)

if conf.val_interval > 0
    for i=1:length(data_debug.im_mat)
        [~,fig_ax{i}] = bia.plot.fig(sprintf('im:%d',i), [1], 1, 0, 1, 1);
    end
end
[fig_loss,ax_loss] = bia.plot.fig('Training Progress', [1 2], 1, 0, 1, 1);
drawnow

%% making tran/val data
cache_train = fullfile(train_dir,'cache_train.mat');
if conf.use_cache && exist(cache_train, 'file')
    fprintf('loaded train roidb: %s\n', cache_train);
    load(cache_train, 'image_roidb_train', 'bbox_means', 'bbox_stds')
else
    fprintf('creating train roidb:');
    [image_roidb_train, bbox_means, bbox_stds] = cpn.seg.roidb(conf, roidb_train);
    if conf.use_cache;  save(cache_train, 'image_roidb_train', 'bbox_means', 'bbox_stds');  end
end
if conf.do_val
    cache_val = fullfile(train_dir,'cache_val.mat');
    if conf.use_cache && exist(cache_val, 'file')
        fprintf('loaded val roidb: %s\n', cache_val);
        load(cache_val, 'image_roidb_val')
    else
        fprintf('creating val roidb:');
        [image_roidb_val] = cpn.seg.roidb(conf, data_val.roidb, bbox_means, bbox_stds);
        if conf.use_cache;  save(cache_val, 'image_roidb_val'); end
    end
    shuffled_inds_val = generate_random_minibatch([], image_roidb_val);
    shuffled_inds_val = shuffled_inds_val(randperm(length(shuffled_inds_val), min(length(shuffled_inds_val))));
end
fprintf('\n');
%% training
shuffled_inds = [];
train_results = [];
iter_ = caffe_solver.iter()+iter_prev_run;
%max_iter = caffe_solver.max_iter()+iter_prev_run;
max_iter = conf.max_iter+iter_prev_run;


if conf.mode == 1
    leg = {{'accuracy_fg', 'accuracy_bg'}, {'loss_cls'}};
else
    leg = {{'accuracy_fg', 'accuracy_bg'}, {'loss_bbox', 'loss_cls'}};
end
train_progress = [];
val_progress = [];
if ismember(conf.mode, 2)
    bbox_stds = bbox_stds([1 4],:);
    bbox_means = bbox_means([1 4],:);
end

input_names = caffe_solver.net.inputs;
output_names = caffe_solver.net.outputs;
num_inputs = length(input_names);


poolobj = gcp('nocreate');
delete(poolobj);
t_start_train = tic;
while (iter_ < max_iter)
    caffe_solver.net.set_phase('train');
    % generate minibatch training data
    [shuffled_inds, sub_db_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train);
    [net_inputs, valid] = get_net_inputs(conf, image_roidb_train(sub_db_inds));
    if ~valid
        shuffled_inds(shuffled_inds == length(image_roidb_train)) = [];
        image_roidb_train(sub_db_inds) = [];
        continue;   
    end
    if num_inputs == 3;        net_inputs(end) = [];    end
    caffe_solver.net.reshape_as_input(net_inputs);
    
    % one iter SGD update
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);
    iter_ = caffe_solver.iter()+iter_prev_run;

    rst = caffe_solver.net.get_output();
    rst = check_error(rst, caffe_solver);
    train_results = cpn.utils.parse_rst(train_results, rst);
    if ~mod(iter_, conf.val_interval)% do valdiation per val_interval iterations
        if conf.do_val
            val_results = do_validation(conf, caffe_solver, image_roidb_val, shuffled_inds_val);
        else
            val_results = [];
        end
        [train_progress, val_progress] = cpn.utils.plot_loss(train_progress, val_progress, train_results, val_results, ax_loss, leg);
        show_state(conf, iter_, train_results, val_results, t_start_train);
        train_results = [];% reset train errors for next interval
        
        caffe_solver.net.set_phase('test');
        for i=1:length(data_debug.im_mat)
            im = data_debug.im_mat{i};
            rois = data_debug.rois{i};% corners
            if isempty(rois);   continue;   end
            stats = get_seg_test(conf, caffe_solver.net, im, rois);
            cla(fig_ax{i})
            imshow(bia.draw.boundary(struct('alpha',0.75), im, stats), 'parent',fig_ax{i});
        end
        caffe_solver.net.set_phase('train');
        drawnow
        diary; diary; % flush diary
    end
    
    if ~mod(iter_, conf.snapshot_interval)
        cpn.io.snapshot(conf, caffe_solver.net, fullfile(train_dir, sprintf('iter_%d%s', iter_, conf.paths.id)), bbox_means, bbox_stds, 'bbox_pred');
    end
end

if conf.deform > 0
    cpn.io.imread();
    cpn.io.gtread();
end
% final snapshot
if rem(iter_, conf.snapshot_interval)
    cpn.io.snapshot(conf, caffe_solver.net, fullfile(train_dir, sprintf('iter_%d%s', iter_, conf.paths.id)), bbox_means, bbox_stds, 'bbox_pred');
end
cpn.io.snapshot(conf, caffe_solver.net, path_final, bbox_means, bbox_stds, 'bbox_pred');
saveas(fig_loss, bia.save.prevent_overwrite(fullfile(train_dir, 'train_error.png')))
save(fullfile(train_dir, 'training_loss.mat'), 'train_results', 'val_results')

copyfile(path_solver, path_solver_final);
copyfile(path_def_test, path_def_test_final);
copyfile(path_def_train, path_def_train_final);

bia.caffe.clear;
fprintf('Training done: %1.2f hrs\n', toc(t_start)/60/60)
diary off;
end


function stats = get_seg_test(conf, net, im, rois)
rois(:,1:4) = bia.convert.bb(rois(:,1:4),'b2c');
rois(:, 1:4) = cpn.seg.adjust_rois(conf, rois(:, 1:4), size(im));
rois_blob = rois;


num_rois = size(rois,1);
rois_blob(:, 5) = [];
rois_blob = [ones(num_rois,1), rois_blob];
if conf.channels == 3 && size(im,3) == 1
   im = repmat(im, [1 1 3]) ;
end
im_blob = single(im) - conf.image_means;
im_blob = permute(im_blob, [2, 1, 3, 4]);

rois_blob = round(rois_blob) - 1; % to c's index (start from 0)
rois_blob = single(permute(rois_blob, [3, 4, 2, 1]));

mask_blob = zeros(num_rois, conf.mask_sz^2);
mask_blob = single(permute(mask_blob, [3, 4, 2, 1]));
net_inputs = {im_blob, rois_blob, mask_blob, mask_blob};

net.reshape_as_input(net_inputs);
net.set_input_data(net_inputs);
net.forward(net_inputs);

seg_score = net.blobs('seg_score').get_data();
seg_exp = exp(seg_score);
segv = squeeze(seg_exp(:,:,2,:)./(sum(seg_exp,3)));
segv = permute(segv, [2 1]);
stats = rois2stats(conf, im, rois, segv, 0);
end
% feats
function stats = rois2stats(conf, im, rois, segv, add_padd)
thresh = 0.5;
mask_sz = conf.mask_sz;
R = size(rois,1);% # of rois
sz_orig = [size(im, 1), size(im, 2)];
rect_crop = round(rois(:,[2 4 1 3]));% rois=[xmin ymin xmax ymax] -> rect_crop=[ymin ymax xmin xmax]
rect_crop(:,[1,3])= max(1, rect_crop(:,[1,3]));
rect_crop(:,[2,4])= min(repmat([size(im, 1), size(im, 2)], size(rect_crop, 1), 1), rect_crop(:,[2,4]));

feats = zeros(R,1);
func = @(a,b,c,d,e,f) cpn_roi_stats(a, b, c, d, e);

for j = 1:R% to speed up imresize
    px = reshape(segv(j, :), mask_sz, mask_sz)';% transpose
    r = rect_crop(j, :);
    bsz = [r(2)-r(1)+1, r(4)-r(3)+1];
    px = imresize(px, bsz, 'bicubic');
    px_bw = keep_largest_region(px > thresh);
    stats(j,1) = func(px_bw, r, sz_orig, add_padd, rois(j,5), feats(j,:));
end
end

function stats = cpn_roi_stats(mask, roi, sz, pad, score, feats)
idx     = find(mask);
[r,c]   = ind2sub(size(mask), idx);
r       = double(r+roi(1)-1-pad);
c       = double(c+roi(3)-1-pad);
idx_rm  = r < 1 | c < 1 | r > sz(1) | c > sz(2);
r(idx_rm) = [];
c(idx_rm) = [];

stats.PixelIdxList = sub2ind(sz, r, c);
stats.Area = length(stats.PixelIdxList);

% code snippets copied from "regionprops"
num_dims = numel(sz);
list = [c, r];
if (isempty(list))
    stats.BoundingBox = [0.5*ones(1,num_dims) zeros(1,num_dims)];
else
    min_corner = min(list,[],1) - 0.5;
    max_corner = max(list,[],1) + 0.5;
    stats.BoundingBox = [min_corner (max_corner - min_corner)];
end
stats.Centroid = mean([r,c],1);
stats.Score = score;
if nargin >= 6
    stats.Features = feats;
end
end


function bw = keep_largest_region(bw)
stats = regionprops(logical(bw), 'Area', 'PixelIdxList');
areas = [stats(:).Area];
[~, idx] = max(areas);
for i=1:length(stats)
    if i~= idx
        bw(stats(i).PixelIdxList) = 0;
    end
end
bw = imfill(bw, 'holes');
end


function [net_inputs, valid] = get_net_inputs(conf, image_roidb_train)
[im_blob, rois_blob, ~, mask_blob, bbox_loss_weights_blob] = cpn_seg_minibatch(conf, image_roidb_train);
net_inputs = {im_blob, rois_blob, mask_blob, bbox_loss_weights_blob};
valid = true;
for i=1:length(net_inputs)
    if isempty(net_inputs{i})
        valid = false;
    end
end

if 0
    rois = squeeze(rois_blob)' + 1;%0 to 1-based indexing
    im = conf.imread(image_roidb_train.image_id);
    im = bia.prep.norm(im,'sqrt');
    mask_vec = squeeze(mask_blob)';
    weight_vec = squeeze(bbox_loss_weights_blob)';
    mask = zeros(image_roidb_train.im_size);
    for i=1:size(rois,1)
        r = rois(i,[3 2 5 4]);
        v = reshape(mask_vec(i,:), conf.mask_sz, conf.mask_sz)';
        w = reshape(weight_vec(i,:), conf.mask_sz, conf.mask_sz)';
        mask(r(1):r(3),r(2):r(4)) = imresize(v, [r([3 4])-r([1 2])]+[1 1], 'nearest');
        weight(r(1):r(3),r(2):r(4)) = imresize(w, [r([3 4])-r([1 2])]+[1 1], 'nearest');
    end
    figure(41)
    imshow(bia.draw.boundary(struct('alpha',1),im,mask),[])
    bia.plot.bb([], bia.convert.bb(rois(:,2:5),'c2m'))
    figure(42)
    imshow(weight,[])
end
if isfield(conf, 'entropy') && conf.entropy;    net_inputs(end) = [];   end
end


function rst = check_error(rst, caffe_solver)
seg_score = caffe_solver.net.blobs('seg_score').get_data();
if sum(strcmp(caffe_solver.net.layer_names, 'rs_seg_labels'))
    labels = squeeze(caffe_solver.net.blobs('rs_seg_labels').get_data());
end
if sum(strcmp(caffe_solver.net.layer_names, 'rs_seg_weights'))
    labels_weights = squeeze(caffe_solver.net.blobs('rs_seg_weights').get_data());
else
    labels_weights = ones(size(labels));
end
if length(size(seg_score)) > 2
    [~, labels_fg] = max(seg_score, [], 3);
    labels_fg = squeeze(labels_fg);
else
    [~, labels_fg] = max(seg_score, [], 1);
end
labels_fg = labels_fg-1;% bg class has label '0'

accurate_fg = (labels_fg == labels) & (labels > 0);
accurate_bg = (labels_fg == 0) & (labels == 0);
accurate = accurate_fg | accurate_bg;
% accuracy_fg = sum(accurate_fg(:) .* labels_weights(:)) / (sum(labels_weights(labels == 1)) + eps);
% accuracy_bg = sum(accurate_bg(:) .* labels_weights(:)) / (sum(labels_weights(labels == 0)) + eps);
accuracy_fg = sum(accurate_fg(:) .* labels_weights(:)) / (sum(labels_weights(labels > 0)) + eps);
accuracy_bg = sum(accurate_bg(:) .* labels_weights(:)) / (sum(labels_weights(labels == 0)) + eps);

rst(end+1) = struct('blob_name', 'accuracy_fg', 'data', accuracy_fg);
rst(end+1) = struct('blob_name', 'accuracy_bg', 'data', accuracy_bg);
end


function val_results = do_validation(conf, caffe_solver, image_roidb_val, shuffled_inds_val)
val_results_loc = [];

caffe_solver.net.set_phase('test');
val_loss_bbox_loc   = [];
val_loss_cls_loc    = [];
val_accuracy_loc    = [];
val_accuracy_fg_loc = [];
val_accuracy_bg_loc = [];
for i = 1:length(shuffled_inds_val)
    sub_db_inds = shuffled_inds_val(i);
    [net_inputs, valid] = get_net_inputs(conf, image_roidb_val(sub_db_inds));
    if ~valid;  continue;   end
    caffe_solver.net.reshape_as_input(net_inputs);
    caffe_solver.net.forward(net_inputs);
    rst = caffe_solver.net.get_output();
    rst = check_error(rst, caffe_solver);
    val_results_loc = cpn.utils.parse_rst(val_results_loc, rst);
    if ismember(conf.mode, 2)
        val_loss_bbox_loc = [val_loss_bbox_loc; mean(val_results_loc.loss_bbox.data)];
    end
    val_loss_cls_loc  = [val_loss_cls_loc;  mean(val_results_loc.loss_cls.data)];
    val_accuracy_loc  = [val_accuracy_loc; 1 - mean(val_results_loc.accuracy.data)];
    val_accuracy_fg_loc  = [val_accuracy_fg_loc; mean(val_results_loc.accuracy_fg.data)];
    val_accuracy_bg_loc  = [val_accuracy_bg_loc; mean(val_results_loc.accuracy_bg.data)];
end
if ismember(conf.mode, 2)
    val_results.loss_bbox.data = val_loss_bbox_loc;
end
val_results.loss_cls.data  = val_loss_cls_loc;
val_results.accuracy.data  = val_accuracy_loc;
val_results.accuracy_fg.data  = val_accuracy_fg_loc;
val_results.accuracy_bg.data  = val_accuracy_bg_loc;
end


function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train)
if isempty(shuffled_inds)
    shuffled_inds = randperm(length(image_roidb_train));
end
if nargout > 1
    sub_inds = shuffled_inds(1);
    shuffled_inds(1) = [];
end
end


function check_gpu_memory(conf, caffe_solver, data_train, data_val, do_val)
szs_train = cell2mat(arrayfun(@(x) x.sz, data_train.roidb, 'UniformOutput', false));
[~,idx] = max(prod(szs_train,2));
max_sz_train = szs_train(idx, :);
check_gpu_memory_loc(conf, caffe_solver, max_sz_train);

if do_val && ~isempty(data_val)
    szs_val = cell2mat(arrayfun(@(x) x.sz, data_val.roidb, 'UniformOutput', false));
    [~,idx] = max(prod(szs_val,2));
    max_sz_val = szs_val(idx, :);
    check_gpu_memory_loc(conf, caffe_solver, max_sz_val);
end
end


function check_gpu_memory_loc(conf, caffe_solver, max_sz)
%  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough
im_blob = single(zeros([max_sz, conf.channels, 1]));% generate pseudo training data with max size
rois_blob = single(repmat([0; 0; 0; min(max_sz)-1; min(max_sz)-1], 1, conf.batch_size));
rois_blob = permute(rois_blob, [3, 4, 1, 2]);

seg_targets_blob = zeros(conf.mask_sz^2, conf.batch_size, 'single');
seg_targets_blob = single(permute(seg_targets_blob, [3, 4, 1, 2]));
seg_loss_weights_blob = seg_targets_blob;

net_inputs = {im_blob, rois_blob, seg_targets_blob, seg_loss_weights_blob};

caffe_solver.net.reshape_as_input(net_inputs);% Reshape net's input blobs
caffe_solver.net.set_input_data(net_inputs);
caffe_solver.step(1);% one iter SGD update

% if 1% use the same net with train to save memory
%     caffe_solver.net.set_phase('test');
%     caffe_solver.net.forward(net_inputs);
%     caffe_solver.net.set_phase('train');
% end
end


function show_state(conf, iter, train_results, val_results, t_start)
if exist('val_results', 'var') && ~isempty(val_results)
    val_results.loss_bbox.data = 1;
end
train_results.loss_bbox.data = 1;

train = struct('err_fg',1 - mean(train_results.accuracy_fg.data), 'err_bg', 1 - mean(train_results.accuracy_bg.data),...
    'loss_cls',mean(train_results.loss_cls.data));
fprintf('Iter:%6d, ', iter);
if exist('val_results', 'var') && ~isempty(val_results)
    val = struct('err_fg',1 - mean(val_results.accuracy_fg.data), 'err_bg', 1 - mean(val_results.accuracy_bg.data),...
        'loss_cls',mean(val_results.loss_cls.data));
    fprintf('train(val): err [fg: %.5f(%.5f), bg: %.5f(%.5f)], loss-cls: %.5f(%.5f), time:%1.1fm\n', ...
        train.err_fg, val.err_fg, train.err_bg, val.err_bg, train.loss_cls, val.loss_cls, toc(t_start)/60);
else
    fprintf('train:: err [fg: %.5f, bg: %.5f], loss-cls: %.5f, time:%1.1fm\n', ...
        train.err_fg, train.err_bg, train.loss_cls, toc(t_start)/60);
end
end


function [im_blob, rois_blob, labels_blob, mask_blob, mask_loss_blob] = cpn_seg_minibatch(conf, image_roidb)

num_images = length(image_roidb);
rois_per_image = conf.batch_size / num_images;
fg_rois_per_image = round(rois_per_image * conf.fg_fraction);

% Get the input image blob
im_blob = single(conf.imread(image_roidb.image_id)) - conf.image_means;
if conf.channels == 3 && size(im_blob,3) == 1
    im_blob = repmat(im_blob, [1 1 3]);
end
% build the region of interest and label blobs
[labels_blob, ~, im_rois, mask_blob, mask_loss_blob] = sample_rois(conf, image_roidb, fg_rois_per_image, rois_per_image);

if isfield(conf, 'roi_old') && conf.roi_old
    im_rois(:, 1:4) = cpn.seg.adjust_rois(conf, im_rois(:, 1:4), size(im_blob));
end

% Add to ROIs blob
%feat_rois = fast_rcnn_map_im_rois_to_feat_rois(conf, im_rois, im_scales(i));
feat_rois = round(im_rois);
batch_ind = ones(size(feat_rois, 1), 1);
rois_blob = [batch_ind, feat_rois];

% fprintf('#Rois:%d, #BG:%d, #FG:%d\n', size(rois_blob,1), sum(labels_blob==0), sum(labels_blob>0));
% permute data into caffe c++ memory, thus [num, channels, height, width]
if length(size(im_blob)) == 3%rgb image
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
end
im_blob = single(permute(im_blob, [2, 1, 3, 4]));
rois_blob = rois_blob - 1; % to c's index (start from 0)
rois_blob = single(permute(rois_blob, [3, 4, 2, 1]));
labels_blob = single(permute(labels_blob, [3, 4, 2, 1]));
mask_blob = single(permute(mask_blob, [3, 4, 2, 1]));
mask_loss_blob = single(permute(mask_loss_blob, [3, 4, 2, 1]));

% if isempty(im_blob);    warning('Empty image: %s', image_roidb.image_id); end
% if isempty(rois_blob);    warning('no rois found in: %s', image_roidb.image_id); end
% if isempty(mask_blob) || isempty(mask_loss_blob);    warning('empty mask found in: %s', image_roidb.image_id); end
end


%% Generate a random sample of ROIs comprising foreground and background examples.
function [labels, overlaps, rois, bbox_targets, bbox_loss_weights] = sample_rois(conf, image_roidb, fg_rois_per_image, rois_per_image)
[overlaps, labels] = max(image_roidb(1).overlap, [], 2);
rois = image_roidb(1).rois;
fg_inds = find(image_roidb.labels == 1);
bg_inds = [];

% Guard against the case when an image has fewer than fg_rois_per_image foreground ROIs
fg_rois_per_this_image = min(fg_rois_per_image, length(fg_inds));
if ~isempty(fg_inds)% Sample foreground regions without replacement
    fg_inds = fg_inds(randperm(length(fg_inds), fg_rois_per_this_image));
end

% Compute number of background ROIs to take from this image (guarding against there being fewer than desired)
bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image;
bg_rois_per_this_image = min(bg_rois_per_this_image, length(bg_inds));
if ~isempty(bg_inds)% Sample foreground regions without replacement
    bg_inds = bg_inds(randperm(length(bg_inds), bg_rois_per_this_image));
end

%keep_inds = [fg_inds; bg_inds];% The indices that we're selecting (both fg and bg)
keep_inds = fg_inds;

% Select sampled values from various arrays
labels = labels(keep_inds);
% Clamp labels for the background ROIs to 0
labels((fg_rois_per_this_image+1):end) = 0;
labels(labels>0) = 1;
overlaps = overlaps(keep_inds);
rois = rois(keep_inds, :);

%     assert(all(labels == image_roidb.bbox_targets(keep_inds, 1)));
% Infer number of classes from the number of columns in gt_overlaps
num_classes = size(image_roidb(1).overlap, 2);
if ismember(conf.mode, 1)
    bbox_targets = image_roidb.seg_masks(keep_inds, :);
    bbox_loss_weights = image_roidb.seg_weights(keep_inds, :);
    if ismember(conf.pos_criteria, 13)
        bbox_targets(ismember(bg_inds, keep_inds), :) = 0;
    end
elseif ismember(conf.mode, 2)
    [bbox_targets, bbox_loss_weights] = get_bbox_regression_labels(conf, ...
        image_roidb.bbox_targets(keep_inds, :), num_classes);
    assert(~sum(image_roidb.bbox_targets(keep_inds, 1) == -1))
    if sum(isnan(image_roidb.bbox_targets(:)))
        sum(isnan(image_roidb.bbox_targets(:)))
    end
end
if 0% for debugging
    invalid_inds = find(image_roidb.bbox_targets(:,1) == -1);
    fig('Debug 2nd stage')
    im = conf.imread(image_roidb(1).image_id);
    imshow(bia.prep.norm(sqrt(single(im))))
    bia.plot.centroids([],markers)
    num_show = min([30, length(fg_inds)]);
    bia.plot.bb([],bia.convert.bb(rois_all(fg_inds(1:num_show), :),'c2m'), 'g')
    num_show = min([30, length(bg_inds)]);
    bia.plot.bb([],bia.convert.bb(rois_all(bg_inds(1:num_show), :),'c2m'), 'r')
    num_show = min([30, length(invalid_inds)]);
    bia.plot.bb([],bia.convert.bb(rois_all(invalid_inds(1:num_show), :),'c2m'), 'y')
    drawnow
end
end


function [bbox_targets, bbox_loss_weights] = get_bbox_regression_labels(conf, bbox_target_data, num_classes)
%% Bounding-box regression targets are stored in a compact form in the roidb.
% This function expands those targets into the 4-of-4*(num_classes+1) representation used
% by the network (i.e. only one class has non-zero targets).
% The loss weights are similarly expanded.
% Return (N, (num_classes+1) * 4, 1, 1) blob of regression targets
% Return (N, (num_classes+1 * 4, 1, 1) blob of loss weights
clss = bbox_target_data(:, 1);
bbox_targets = zeros(length(clss), 4 * (num_classes+1), 'single');
bbox_loss_weights = zeros(size(bbox_targets), 'single');
inds = find(clss > 0);
for i = 1:length(inds)
    ind = inds(i);
    cls = clss(ind);
    targets_inds = (1+cls*4):((cls+1)*4);
    bbox_targets(ind, targets_inds) = bbox_target_data(ind, 2:end);
    bbox_loss_weights(ind, targets_inds) = 1;
end
end
