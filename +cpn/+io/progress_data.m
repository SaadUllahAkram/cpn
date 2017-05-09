function [dataset_train, dataset_val, dataset_debug, do_val, conf_cpn, conf_seg] = progress_data(opts, conf_cpn, conf_seg, do_val, imdb)
% Returns data (validation/display) which can be used for tracking training progress
%

do_seg = ~isempty(conf_seg);
opts_default = struct('num_test_max',1,'num_train_max',1,'max_h',700);
opts = bia.utils.updatefields(opts_default, opts);

max_h = opts.max_h;

n_train = length(imdb.roidb_train);
n_test = length(imdb.roidb_val);

num_test_max = min(opts.num_test_max, n_test);
num_train_max = min(opts.num_train_max, n_train);

if num_train_max
    idx_train_ims = randperm(n_train, num_train_max);
    for i=1:num_train_max
        idx = idx_train_ims(i);
        dataset_debug.im_names{i,1} = [imdb.roidb_train(idx).image_id, '.png'];
        dataset_debug.im_labels{i,1} = unique(imdb.roidb_train(idx).class);
        if do_seg
            dataset_debug.im_mat{i,1} = conf_seg.imread([imdb.roidb_train(idx).image_id, '.png']);
        else
            dataset_debug.im_mat{i,1} = conf_cpn.imread([imdb.roidb_train(idx).image_id, '.png']);
        end
        dataset_debug.im_mat{i,1} = dataset_debug.im_mat{i}(1:min(size(dataset_debug.im_mat{i}, 1), max_h),:,:);
    end
end

if num_test_max
    idx_test_ims = randperm(n_test, num_test_max);
    for k=1:num_test_max
        idx = idx_test_ims(k);
        kk = num_train_max+k;
        dataset_debug.im_names{kk} = [imdb.roidb_val(idx).image_id, '.png'];
        dataset_debug.im_labels{kk} = unique(imdb.roidb_val(idx).class);
        dataset_debug.im_mat{kk} = conf_cpn.imread([imdb.roidb_val(idx).image_id, '.png']);
        dataset_debug.im_mat{kk} = dataset_debug.im_mat{kk}(1:min(size(dataset_debug.im_mat{kk}, 1), max_h),:,:);
    end
else
    do_val = false;
end

dataset_train = struct('roidb',imdb.roidb_train);
if do_val
    dataset_val.roidb = imdb.roidb_val;
else
    dataset_val.roidb = [];
end
dataset_debug.cpn  = struct('num_props', 400, 'score_thresh', 0.5, 'pause_t', 0.1, 'use_pruned_props', 1);
dataset_debug.nms  = struct('nms_overlap_thres', 0.2, 'after_nms_topN', 900);

n_progress = length(dataset_debug.im_mat);
fprintf('Train images:%d, Val images: %d, Display images:%d\n',n_train,n_test,n_progress)

if do_seg
    conf_seg.cpn_dir = conf_cpn.paths.dir;
    cache_rois = fullfile(conf_seg.paths.dir,'cache_rois.mat');
    if ~(conf_cpn.use_cache && conf_seg.use_cache)
        if exist(cache_rois, 'file')
            delete(cache_rois)
        end
    end
    if ~exist(cache_rois, 'file')
        [dataset_train, dataset_val] = get_proposed_rois(conf_cpn, dataset_train, dataset_val);
        if conf_cpn.use_cache
            save(cache_rois, 'dataset_train', 'dataset_val')
        end
    else
        load(cache_rois, 'dataset_train', 'dataset_val')
    end
    
    cpn_net = bia.caffe.load(fullfile(conf_cpn.paths.dir, sprintf('final_test%s.prototxt', conf_cpn.paths.id)), fullfile(conf_cpn.paths.dir, sprintf('final%s', conf_cpn.paths.id)));
    for i=1:length(dataset_debug.im_mat)
        boxes = cpn.bb.proposals(conf_cpn, cpn_net, dataset_debug.im_mat(i));
        boxes{1}(boxes{1}(:,5) < 0.9,:) = [];
        dataset_debug.rois(i) = boxes;% top_left bottom_right corners
    end
    bia.caffe.clear
end

end


function [dataset_train, dataset_val] = get_proposed_rois(conf_cpn, dataset_train, dataset_val)
% update roidb to include proposed ROIS
dataset_train.roidb = do_proposal_test(conf_cpn, dataset_train.roidb);
if ~isempty(dataset_val.roidb)
    dataset_val.roidb = do_proposal_test(conf_cpn, dataset_val.roidb);
end
end


function roidb_new = do_proposal_test(conf, roidb)
boxes = proposal_test(conf, roidb);
roidb_new = cpn.seg.roidb_from_proposal(roidb, boxes, 'keep_raw_proposal', false);
end


function aboxes = proposal_test(conf, roidb)
% caffe_log_file_base = fullfile(conf.paths.dir, 'caffe_log_seg_props');
% caffe.init_log(caffe_log_file_base);
cpn_net = bia.caffe.load(fullfile(conf.paths.dir, sprintf('final_test%s.prototxt', conf.paths.id)), fullfile(conf.paths.dir, sprintf('final%s', conf.paths.id)));

num_images = length(roidb);
% all detections are collected into: all_boxes[image] = N x 5 array of detections in % (x1, y1, x2, y2, score)
aboxes = cell(num_images, 1);

disp_iter = max(1, round(num_images/10));
time_str = '';
fprintf('computing rois:%d::', num_images)
conf.nms.nms_overlap_thres = conf.train_seg_nms;
onms = conf.nms;
onms.after_nms_topN = 1000;
bia.print.struct(onms);

time_start = tic;
for i = 1:num_images
    if rem(i, disp_iter) == 0
        fprintf(repmat('\b', 1, length(time_str)))
        time_str = sprintf('%3.0f%%: %4.1fsec', 100*i/num_images, toc(time_start));
        fprintf('%s', time_str)
    end
    im = conf.imread(roidb(i).image_id);
    aboxes(i) = cpn.bb.proposals(conf, cpn_net, im);
    assert(~isempty(aboxes{i}))
    if 0
        imshow(im)
        bia.plot.bb([], bia.convert.bb(aboxes{i}(1:30,1:4),'c2m'))
        drawnow
    end
end
fprintf('\n')
bia.caffe.clear
end