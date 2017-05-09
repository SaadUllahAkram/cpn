function imdb = get_imdb(conf)
%
% Inputs:
%     dataset_name: can be a single dataset or a collection of datasets
%     seq : [1,2] (selects that sequence) or "0" selects both sequences
%     type : what type of data to load
% Outputs:
%     imdb_file : name of imdb file for training
%

root_cpn_data = conf.paths.imdb;
cpn.io.imread()
cpn.io.gtread()

imdb_path = fullfile(root_cpn_data, sprintf('%s-%02d_gt%d_flips%d_rot%d_maxsize%d.mat', conf.dataset, conf.train_seq, conf.gt_version, conf.use_flips, conf.rotations_interval, conf.max_dim));
if ~conf.use_cache && exist(imdb_path, 'file')
    delete(imdb_path)
end
if ~exist(imdb_path,'file')
    imdb = prepare_imdb(conf, root_cpn_data);
    if conf.test_loss
        test_seq = cpn.seq(conf.train_seq, 1);
        conf_test = bia.utils.setfields(conf, 'train_seq', test_seq, 'use_flips', 0, 'rotations_interval', 0);
        imdb_test = prepare_imdb(conf_test, root_cpn_data);
        imdb.roidb_val = imdb_test.roidb_train;
    end
    save(imdb_path, 'imdb')
else
    warning('Loading old imdb')
    load(imdb_path, 'imdb')
end

end


function imdb = prepare_imdb(conf, image_dir)
dataset_id = conf.dataset;
class_num = 1;
fprintf('#################\n%s:: Class id:%d, Rescale: %1.2f\n', dataset_id, class_num, conf.scale)
% gather file_name
names_all = {};
tlists = [];
versions = sprintf('gtv%02d_imv%02d_scale%1.2f_c%d', conf.gt_version, conf.im_version, conf.scale, conf.channels);
if conf.train_seq == 0
    train_seq = [1 2];
else
    train_seq = conf.train_seq;
end
for s = train_seq
    seq_name = sprintf('%s-%02d', dataset_id, s);
    gt = bia.datasets.load(seq_name, {'gt'}, struct('scale',conf.scale, 'version',[conf.gt_version,conf.im_version,0]));
    if strcmp(conf.stage, 'bb')% only use fully segmented frames
        tlist = gt.seg.info(gt.seg.info(:,3)==1,1);
    elseif strcmp(conf.stage, 'seg')% use partially segmented frames as well
        tlist = gt.seg.info(gt.seg.info(:,2)==1, 1);
    end
    N = (1 + 3*(conf.use_flips==1))*length(tlist);
    if conf.rotations_interval% rot_interval
        rot_angles = 0:conf.rotations_interval:359;
    else
        rot_angles = [];
    end
    if conf.use_flips
        rot_angles = unique([rot_angles, 90]);
    end
    num_rots = length(rot_angles);
    o_prev = length(names_all);
    names_all = [names_all; cell(N*(num_rots+1),1)];% due to flips
    tlists = [tlists; [s*ones(length(tlist),1), tlist]];
    for i=1:length(tlist)% get orig names+flipped names
        o = o_prev+(i-1)*(1 + 3*(conf.use_flips==1));
        cur_name = sprintf('%s-%02d-t%03d_%s', dataset_id, s, tlist(i), versions);
        if conf.use_flips
            fun_names = @(x) {x; sprintf('%s_fliplr',x); sprintf('%s_flipud',x); sprintf('%s_flipboth',x)};
            names_all(o+1:o+4,1) = fun_names(cur_name);
        else
            fun_names = @(x) {x};
            names_all(o+1,1) = fun_names(cur_name);
        end
    end
    for i = 1:N% add rot names
        j = o_prev+i;
        o = o_prev+N + (i-1)*num_rots;
        names_all(o+1:o+num_rots,1) = arrayfun(@(x) sprintf('%s_rot%03d', names_all{j}, x), rot_angles, 'UniformOutput', false);
    end
end
% get bboxes, sizes, class, names, [im, mask, invalid]
k = 0;
debug_show = 0;
for i=1:length(names_all)
    names = split_im(names_all{i}, conf.max_dim);
    for j=1:length(names)
        k = k+1;
        [invalid, bbox_tl_br] = cpn.io.gtread(names{j});
        t = sscanf(names{j},[seq_name,'-t%d']);
        data(k,1) = struct('name',names{j},'class',class_num,'bbox',bbox_tl_br,'sz',size(invalid),'dataset',seq_name,'t',t);
        if debug_show
            im = cpn.io.imread(names{j});
            im = repmat(im(:,:,1), [1 1 3]);% only retains 1st channel even for rgb
            im(:,:,2) = ~invalid.*single(im(:,:,1));
            imshow(im,[]);
            bia.plot.bb([], bia.convert.bb(data(k).bbox,'c2m'));
            drawnow
        end
    end
end
opts_images = struct('max_num_classes',1,'image_dir',image_dir,'used_ims',tlists,'val_ratio',conf.val_ratio);
imdb = create_imdb(data, opts_images);

end


function imdb = create_imdb(data, opts)
% saves the data for CPN training
% Input:
%     data.{}
%     % data.im{1:N} -> images
%     % data.mask{1:N} -> corresponding masks
%     % data.invalid{1:N} -> invalid region
%     % data.names -> file name identifying dataset, seq #, t, rot, flip, etc
%     % data.class -> class label
%     % data.max_im_dim -> max size of all training images
%     opts: options
%

opts_default    = struct('save_mask', 0, 'val_ratio', 1, 'max_num_classes', 20, 'min_test_im', 0, 'save_png', 0, 'save_im_mat', 0);
opts            = bia.utils.updatefields(opts_default, opts);

image_dir       = opts.image_dir;
save_png        = opts.save_png;
save_im_mat     = opts.save_im_mat;
save_mask       = opts.save_mask;
disp_bboxes     = 0;

val_ratio       = opts.val_ratio;
min_test_im     = opts.min_test_im;% at least this many ims are kept for testing
max_num_classes = opts.max_num_classes;

num_samples     = length(data);
disp_iter       = max(1, round(num_samples/50));

bia.save.mkdir(image_dir)

max_im_dim = -1;

idx_rm = [];
rois(num_samples,1) = struct('image_id',{-1},'sz',{-1},'gt',{-1},'overlap',{-1},'class',{-1},'boxes',{-1});
for i=1:num_samples% loop over images
    im_name = [data(i).name, '.png'];
    sz = data(i).sz(1:2);
    class = data(i).class;
    bbox_tl_br = data(i).bbox;
    
    max_im_dim = max(max(sz), max_im_dim);
    num_gt = size(bbox_tl_br, 1);
    
    overlap = zeros(num_gt, max_num_classes);
    overlap(:,class) = 1;
    
    sizes(i,:) = sz;
    rois(i,1) = struct('image_id',im_name,...
        'sz',sz,...
        'gt',true(num_gt, 1),...
        'overlap',overlap,...%num_gt x N_classes: overlap with the object
        'class', class*ones(num_gt, 1),...%max_num_classes
        'boxes', bbox_tl_br); % num_gt x 4;% bbox in corner format [x1, y1, x2, y2]
    
    if num_gt == 0
       idx_rm = [idx_rm, i];
    end
    if save_png || save_mask || save_im_mat || disp_bboxes
        im = cpn.io.imread(im_name);
        if disp_bboxes && rem(i, disp_iter) == 0
            ims = imfuse(im, mask, 'falsecolor');
            imshow(ims);
            bia.plot.bb([],stats);
            drawnow
        end
        if save_png
            imwrite(im, fullfile(image_dir, im_name));
        end
        if save_mask
            save(fullfile(image_dir, strrep(im_name, '.png', '_mask.mat')), 'mask');
        end
        if save_im_mat
            assert(isequal(size(im), size(data(i).sz)), 'size miss-match: rescaling error')
            save(fullfile(image_dir, strrep(im_name, '.png', '.mat')), 'im');
        end
    end
end
sizes(idx_rm,:) = [];
rois(idx_rm) = [];
num_samples = length(rois);


% Creating imdb
imdb = init_dataset(image_dir, max_num_classes, max_im_dim, opts.used_ims);

% Splitting in training and testing data
idx_samples = randperm(num_samples);
idx_train = idx_samples(1:round((1-val_ratio)*num_samples));
idx_val = setdiff(idx_samples, idx_train);
if length(idx_val) < min_test_im
    min_sz = sizes(1,:);
    valid_ims = find(cellfun(@(x) isequal(x, min_sz), mat2cell(sizes, ones(size(sizes, 1), 1))));%ims without rotation [can be flipped or 180o rot versions]
    idx_val = valid_ims(randperm(length(valid_ims), min(length(valid_ims), min_test_im)));
    %     idx_train(idx_val) = [];
end
imdb.roidb_val = rois(idx_val);
imdb.roidb_train = rois(idx_train);
% assert(isempty(intersect(idx_train, idx_val)), 'No im should be in both train and val')
fprintf('# Training images: %d, # Validation images: %d\n', length(idx_train), length(idx_val))
end


function imdb = init_dataset(image_dir, num_classes, max_im_dim, used_ims)
dataset_name = 'cpn';
im_extension = 'png';
imdb.opts = struct('name',dataset_name,'image_dir',image_dir,'extension',im_extension,'num_classes',num_classes,'max_dim',max_im_dim,'used_ims',used_ims);
imdb.roidb_val = struct();
end


function [names, num_splits, y_splits, x_splits] = split_im(name, max_sz)
im = cpn.io.imread(name);
sz = size(im);
y_splits = ceil(sz(1)/max_sz);
x_splits = ceil(sz(2)/max_sz);
num_splits = y_splits*x_splits;
if max(sz) > max_sz
    names = arrayfun(@(x) sprintf('%s_imsplit_%d_%d_%d', name, max_sz, num_splits, x), [1:num_splits]', 'UniformOutput', false);
else
    names{1,1} = name;
end
end