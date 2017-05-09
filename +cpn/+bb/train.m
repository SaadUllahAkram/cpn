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

conf_bb = conf;%#ok<NASGU>
save(path_config, 'conf_bb')

roidb_train = data_train.roidb;

timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
caffe.init_log(fullfile(train_dir, 'caffe_'));% adds timestamp in caffe
diary(fullfile(train_dir, sprintf('matlab_%s.txt', timestamp)));

iter_prev_run = 0;
if conf.resume_train% check how far training has progressed so far
    [path_init_file, iter_prev_run, path_solver] = cpn.utils.resume_training(train_dir, path_solver);
end

bia.print.fprintf('Red', 'Training CPN-BB\n')
t_start = tic;
caffe_solver = caffe.Solver(path_solver);
if ~isempty(path_init_file) && exist(path_init_file,'file')
    bia.print.fprintf('red',sprintf('\nResuming training from: %s\n', path_init_file))
    caffe_solver.net.copy_from(path_init_file);
end
% if conf.debug
%     debug_anchors(conf, caffe_solver.net);
% end
if isempty(data_val.roidb); conf.do_val = 0;    end
check_gpu_memory(conf, caffe_solver, data_train, data_val, conf.do_val);

bia.caffe.print_sz(caffe_solver.net);
bia.print.struct(conf,{'dataset'});
fprintf('anchors:\n')
disp(conf.anchors)

cache_train = fullfile(train_dir, 'cache_train_imdb.mat');
if conf.use_cache && exist(cache_train, 'file')
    load(cache_train, 'image_roidb_train', 'bbox_means', 'bbox_stds')
else
    [image_roidb_train, bbox_means, bbox_stds] = cpn.bb.roidb(conf, roidb_train);
    if conf.use_cache;  save(cache_train, 'image_roidb_train', 'bbox_means', 'bbox_stds');  end
end
if conf.do_val
    cache_val = fullfile(train_dir, 'cache_val_imdb.mat');
    if conf.use_cache && exist(cache_val, 'file')
        load(cache_val, 'image_roidb_val')
    else
        [image_roidb_val] = cpn.bb.roidb(conf, data_val.roidb, bbox_means, bbox_stds);
        if conf.use_cache;  save(cache_val, 'image_roidb_val'); end
    end
    shuffled_inds_val = generate_random_minibatch([], image_roidb_val);
end
%% -------------------- Training --------------------
% training
shuffled_inds = [];
train_results = [];
iter_ = caffe_solver.iter()+iter_prev_run;
% max_iter = caffe_solver.max_iter()+iter_prev_run;
max_iter = conf.max_iter+iter_prev_run;

if conf.val_interval > 0
    for i=1:length(data_debug.im_mat)
        [~,fig_ax(i)] = bia.plot.fig(sprintf('im:%d',i),1,1,1,1,1);
    end
end
[fig_loss, ax_loss] = bia.plot.fig('Training Progress',[1 2],1,0,0,1);
leg = {{'accuracy_fg', 'accuracy_bg'}, {'loss_bbox', 'loss_cls'}};
train_progress  = [];
val_progress    = [];
val_results = [];
t_start2 = tic;
n_ims = 10*length(roidb_train);
conf.nms = struct('per_nms_topN',10000,'nms_overlap_thres',0.1,'after_nms_topN',400,'nms_score',0.5,'use_gpu',true);

poolobj = gcp('nocreate');
delete(poolobj);
% input_names = caffe_solver.net.inputs;
% output_names = caffe_solver.net.outputs;
while (iter_ < max_iter)
    if conf.deform > 0
        if iter_ == 0
            %[imdb_deform, roidb_deform] = cpn.exp.deform(struct('max_deform',conf.deform), roidb_train);% deformed images, gt and invalids
            %image_roidb_train = cpn.bb.roidb(conf, roidb_deform, bbox_means, bbox_stds);
        elseif iter_ < n_ims% use orig images
        elseif iter_ == max_iter - n_ims% use orig images
            cpn.exp.deform(struct('reset',1), roidb_train);% deformed images, gt and invalids
            image_roidb_train = cpn.bb.roidb(conf, roidb_deform, bbox_means, bbox_stds);
        elseif rem(iter_, n_ims) == 0% use deformed images
            cpn.exp.deform(struct('reset',1), roidb_train);% deformed images, gt and invalids
            [roidb_deform] = cpn.exp.deform(struct('max_deform',conf.deform), roidb_train);% deformed images, gt and invalids
            image_roidb_train = cpn.bb.roidb(conf, roidb_deform, bbox_means, bbox_stds);
        end
    end
    % generate minibatch training data
    [shuffled_inds, sub_db_inds] = generate_random_minibatch(shuffled_inds, image_roidb_train);
    caffe_solver.net.set_phase('train');
    
    if conf.ohem
        [net_inputs, rst] = cpn.exp.ohem(conf, caffe_solver, image_roidb_train(sub_db_inds));
    else
        net_inputs = cpn.bb.get_minibatch(conf, image_roidb_train(sub_db_inds));
        % proposal_visual_debug(conf, image_roidb_train(sub_db_inds), net_inputs, bbox_means, bbox_stds, conf.classes, scale_inds);
        caffe_solver.net.reshape_as_input(net_inputs);
        caffe_solver.net.set_input_data(net_inputs);
        caffe_solver.step(1);
        rst = caffe_solver.net.get_output();
    end
    rst = check_error(rst, caffe_solver);
    train_results = cpn.utils.parse_rst(train_results, rst);
    % check_loss(rst, caffe_solver, net_inputs);
    if ~mod(iter_, conf.val_interval)% do valdiation per val_interval iterations
        if conf.do_val
            val_results = do_validation(conf, caffe_solver, image_roidb_val, shuffled_inds_val);
        else
            val_results = [];
        end
        [train_progress, val_progress] = cpn.utils.plot_loss(train_progress, val_progress, train_results, val_results, ax_loss, leg);
        show_state(iter_, train_results, val_results, t_start2);
        train_results = [];% reset train errors for next interval
        fprintf(', #samples, pos:%d, neg:%d\n', bia.utils.ssum(net_inputs{2}==1 & net_inputs{3}==1), bia.utils.ssum( net_inputs{2}==0 & net_inputs{3}==1 ) )

        [weights_back, bias_back] = cpn.utils.bb_weights_norm(conf, caffe_solver.net, 'proposal_bbox_pred', bbox_means, bbox_stds);
        caffe_solver.net.set_phase('test');
        bb_nms = get_props_test(conf,caffe_solver.net, data_debug.im_mat, fig_ax);

        caffe_solver.net.set_phase('train');
        cpn.utils.bb_weights_norm(conf, caffe_solver.net, 'proposal_bbox_pred', bbox_means, bbox_stds, weights_back, bias_back);
        
        diary; diary; % flush diary
    end
    
    if ~mod(iter_, conf.snapshot_interval)% snapshot
        cpn.io.snapshot(conf, caffe_solver.net, fullfile(train_dir, sprintf('iter_%d%s', iter_, conf.paths.id)), bbox_means, bbox_stds, 'proposal_bbox_pred');
    end
    iter_ = caffe_solver.iter()+iter_prev_run;
end

if conf.deform > 0
    cpn.io.imread()
    cpn.io.gtread()
end
% final validation
if conf.do_val
    do_validation(conf, caffe_solver, image_roidb_val, shuffled_inds_val);
end

% final snapshot
if rem(iter_, conf.snapshot_interval)
    cpn.io.snapshot(conf, caffe_solver.net, fullfile(train_dir, sprintf('iter_%d%s', iter_, conf.paths.id)), bbox_means, bbox_stds, 'proposal_bbox_pred');
end
cpn.io.snapshot(conf, caffe_solver.net, path_final, bbox_means, bbox_stds, 'proposal_bbox_pred');
saveas(fig_loss, bia.save.prevent_overwrite(fullfile(train_dir, 'train_error.png')))
save(fullfile(train_dir, 'training_loss.mat'), 'train_results', 'val_results')

copyfile(path_solver, path_solver_final);
copyfile(path_def_test, path_def_test_final);
copyfile(path_def_train, path_def_train_final);

bia.caffe.clear;
fprintf('Training done: %1.2f hrs\n', toc(t_start)/60/60)
diary off;
end


function bb = get_props_test(conf, net, ims, axs)
n = length(ims);
bb_nms = cell(n,1);
for i=1:n
    im = ims{i};
    ax = axs(i);
    cla(ax)
    if conf.channels == 3 && size(im,3) == 1
       im = repmat(im, [1 1 3]) ;
    end
    im_blob = single(im) - conf.image_means;
    net_inputs{1} = permute(im_blob, [2, 1, 3, 4]);
    featuremap_size = [conf.output_map(size(net_inputs{1},1)), conf.output_map(size(net_inputs{1},2))];
    net_inputs{2} = zeros([featuremap_size, conf.anchors_num]);
    net_inputs{3} = net_inputs{2};
    net_inputs{4} = zeros([featuremap_size, 4 * conf.anchors_num]);
    net_inputs{5} = net_inputs{4};

    net.reshape_as_input(net_inputs);
    net.set_input_data(net_inputs);
    net.forward(net_inputs);
    box_deltas = net.blobs('proposal_bbox_pred').get_data();
    o1 = box_deltas;
    scores_un = net.blobs('proposal_cls_score_reshape').get_data();% apply softmax on top
    scores_un = exp(scores_un);
    scores = scores_un(:, :, 2)./sum(scores_un,3);

    featuremap_size = [size(box_deltas, 2), size(box_deltas, 1)];
    box_deltas = permute(box_deltas, [3, 2, 1]);
    box_deltas = reshape(box_deltas, 4, [])';
    anchors = cpn.bb.get_anchors(conf, size(im), '', featuremap_size);
    pred_boxes = cpn.utils.fast_rcnn_bbox_transform_inv(anchors, box_deltas);
    pred_boxes = clip_boxes(pred_boxes, size(im,2), size(im,1));

    scores = reshape(scores, size(o1, 1), size(o1, 2), []);
    scores = permute(scores, [3, 2, 1]);
    scores = scores(:);
    [scores, scores_ind] = sort(scores, 'descend');
    boxes = pred_boxes(scores_ind, :);
    bb = cpn.utils.boxes_filter([boxes, scores],conf.nms.per_nms_topN,conf.nms.nms_overlap_thres,conf.nms.after_nms_topN,1);% [xmin, ymin, xmax, ymax]
    bb(:,1:4) = bia.convert.bb(bb(:,1:4), 'c2b');
    bb_nms{i} = bb;
    
    imshow(bia.prep.norm(im),[], 'Parent', ax)
    bb = sortrows(bb, 5);
    cols = bia.utils.colors('scale-redblue', bb(:,5).^2);
    bia.plot.bb(ax, bb(:, 1:4), cols)
end
drawnow
end


function boxes = clip_boxes(boxes, im_width, im_height)
boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);% x1 >= 1 & <= im_width
boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);% y1 >= 1 & <= im_height
boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);% x2 >= 1 & <= im_width
boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);% y2 >= 1 & <= im_height
end


function val_results = do_validation(conf, caffe_solver, image_roidb_val, shuffled_inds_val)
val_results = [];
caffe_solver.net.set_phase('test');
for i = 1:length(shuffled_inds_val)
    net_inputs = cpn.bb.get_minibatch(conf, image_roidb_val(shuffled_inds_val(i)));
    caffe_solver.net.reshape_as_input(net_inputs);% Reshape net's input blobs
    caffe_solver.net.forward(net_inputs);
    rst = caffe_solver.net.get_output();
    rst = check_error(rst, caffe_solver);
    val_results = cpn.utils.parse_rst(val_results, rst);
end
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


function rst = check_error(rst, caffe_solver)
cls_score = caffe_solver.net.blobs('proposal_cls_score_reshape').get_data();
labels = caffe_solver.net.blobs('labels_reshape').get_data();
labels_weights = caffe_solver.net.blobs('labels_weights_reshape').get_data();
%     accurate_fg = (cls_score(:, :, 2) > cls_score(:, :, 1)) & (labels == 1);
%     accurate_bg = (cls_score(:, :, 2) <= cls_score(:, :, 1)) & (labels == 0);
[~, labels_fg] = max(cls_score, [], 3);
labels_fg = labels_fg-1;% bg class has label '0'
accurate_fg = (labels_fg == labels) & (labels > 0);
accurate_bg = (labels_fg == 0) & (labels == 0);
accurate = accurate_fg | accurate_bg;
%     accuracy_fg = sum(accurate_fg(:) .* labels_weights(:)) / (sum(labels_weights(labels == 1)) + eps);
%     accuracy_bg = sum(accurate_bg(:) .* labels_weights(:)) / (sum(labels_weights(labels == 0)) + eps);
accuracy_fg = sum(accurate_fg(:) .* labels_weights(:)) / (sum(labels_weights(labels > 0)) + eps);
accuracy_bg = sum(accurate_bg(:) .* labels_weights(:)) / (sum(labels_weights(labels == 0)) + eps);

rst(end+1) = struct('blob_name', 'accuracy_fg', 'data', accuracy_fg);
rst(end+1) = struct('blob_name', 'accuracy_bg', 'data', accuracy_bg);
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
im_blob = single(zeros([max_sz, conf.channels, 1]));

anchor_num = size(conf.anchors, 1);
output_width = conf.output_map.values({size(im_blob, 1)});
output_width = output_width{1};
output_height = conf.output_map.values({size(im_blob, 2)});
output_height = output_height{1};
labels_blob = single(zeros(output_width, output_height, anchor_num, 1));
labels_weights = labels_blob;
bbox_targets_blob = single(zeros(output_width, output_height, anchor_num*4, 1));
bbox_loss_weights_blob = bbox_targets_blob;

net_inputs = {im_blob, labels_blob, labels_weights, bbox_targets_blob, bbox_loss_weights_blob};

caffe_solver.net.reshape_as_input(net_inputs);% Reshape net's input blobs
caffe_solver.net.set_input_data(net_inputs);
caffe_solver.step(1);% one iter SGD update

% if 1%do_val% use the same net with train to save memory
%     caffe_solver.net.set_phase('test');
%     caffe_solver.net.forward(net_inputs);
%     caffe_solver.net.set_phase('train');
% end
end





function show_state(iter, train_results, val_results, t_start)
train = struct('err_fg',1 - mean(train_results.accuracy_fg.data), 'err_bg', 1 - mean(train_results.accuracy_bg.data),...
    'loss_cls',mean(train_results.loss_cls.data), 'loss_bbox', mean(train_results.loss_bbox.data));
fprintf('iter:%5d,', iter);
if exist('val_results', 'var') && ~isempty(val_results)
    val = struct('err_fg',1 - mean(val_results.accuracy_fg.data), 'err_bg', 1 - mean(val_results.accuracy_bg.data),...
        'loss_cls',mean(val_results.loss_cls.data), 'loss_bbox', mean(val_results.loss_bbox.data));
    fprintf('train(val): err [fg: %.3g(%.3g), bg: %.3g(%.3g)], loss [cls %.3g(%.3g) + reg %.3g(%.3g)], time:%1.1fm', ...
        train.err_fg, val.err_fg, train.err_bg, val.err_bg, train.loss_cls, val.loss_cls, train.loss_bbox, val.loss_bbox, toc(t_start)/60)
else
    fprintf('train: err [fg: %.3g, bg: %.3g], loss [cls %.3g + reg %.3g], time:%1.1fm', ...
        train.err_fg, train.err_bg, train.loss_cls, train.loss_bbox, toc(t_start)/60)
end
end


function check_loss(rst, caffe_solver, input_blobs)
im_blob = input_blobs{1};
labels_blob = input_blobs{2};
label_weights_blob = input_blobs{3};
bbox_targets_blob = input_blobs{4};
bbox_loss_weights_blob = input_blobs{5};

regression_output = caffe_solver.net.blobs('proposal_bbox_pred').get_data();
% smooth l1 loss
regression_delta = abs(regression_output(:) - bbox_targets_blob(:));
regression_delta_l2 = regression_delta < 1;
regression_delta = 0.5 * regression_delta .* regression_delta .* regression_delta_l2 + (regression_delta - 0.5) .* ~regression_delta_l2;
regression_loss = sum(regression_delta.* bbox_loss_weights_blob(:)) / size(regression_output, 1) / size(regression_output, 2);

confidence = caffe_solver.net.blobs('proposal_cls_score_reshape').get_data();
labels = reshape(labels_blob, size(labels_blob, 1), []);
label_weights = reshape(label_weights_blob, size(label_weights_blob, 1), []);

confidence_softmax = bsxfun(@rdivide, exp(confidence), sum(exp(confidence), 3));
confidence_softmax = reshape(confidence_softmax, [], 2);
confidence_loss = confidence_softmax(sub2ind(size(confidence_softmax), 1:size(confidence_softmax, 1), labels(:)' + 1));
confidence_loss = -log(confidence_loss);
confidence_loss = sum(confidence_loss' .* label_weights(:)) / sum(label_weights(:));

results = cpn.utils.parse_rst([], rst);
fprintf('C++   : conf %f, reg %f\n', results.loss_cls.data, results.loss_bbox.data);
fprintf('Matlab: conf %f, reg %f\n', confidence_loss, regression_loss);
end