function test(opts, conf_bb, conf_seg, dataset_names, test_seqs)
% BB
% NMS
% SEG
% NMS
% EVAL

opts = bia.utils.setfields(opts,...
    'train_str',{'','Train-'},'str_00',{'','00'},...
    'str_train',{'Test ', 'Train'});

dataset_names = {dataset_names};
if ismember(dataset_names, {'FHP'})
    dataset_names = {'Hist-BM','PhC-HeLa-Ox','Fluo-N2DL-HeLa'};
    opts = bia.utils.setfields(opts, 'channels',3,'model_str','FHP-');
elseif ismember(dataset_names, {'Hist-BM'})
    opts = bia.utils.setfields(opts, 'channels',3,'model_str','');
else
    opts = bia.utils.setfields(opts, 'channels',1,'model_str','');
end


if isempty(conf_seg)% only cpn-bb evaluation
    assert(strcmp(opts.use_seg, 'gt'))
    opts.only_props = 1;
end

if strcmp(opts.process, 'whole')
    opts_loc = struct('tra',0,'seg',0);
elseif strcmp(opts.process, 'seg')
    opts_loc = struct('tra',0,'seg',1);
elseif strcmp(opts.process, 'seg_fully')
    opts_loc = struct('tra',0,'seg',2);
elseif strcmp(opts.process, 'tra')
    opts_loc = struct('tra',1,'seg',0);
end

if test_seqs == 0 && ~opts.save
    opts.eval_train = 1;
end

for dataset_name = dataset_names
    if strcmp(dataset_name{1}, 'Fluo-N2DL-HeLa');        opts = bia.utils.setfields(opts,'use_sqrt',1,'alpha',0.5,'linewidth',1);
    elseif strcmp(dataset_name{1}, 'Hist-BM');        opts = bia.utils.setfields(opts,'use_sqrt',0,'alpha',1,'linewidth',2);
    else;        opts = bia.utils.setfields(opts,'use_sqrt',0,'alpha',0.5,'linewidth',1);
    end
    for train = opts.eval_train % 0(test eval), 1(train eval)
        clear bb_nms bb_greedy seg_nms seg_greedy
        clearvars -except dataset_name train opts opts_loc model dataset_names test_seqs nms_iou thresh conf_bb conf_seg

        seq_ids = cpn.seq(test_seqs, train);
        if opts.verbose >= 2
            if train == 1;  fprintf('**********Training Run**********\n')
            else;   fprintf('**********Testing Run**********\n')
            end
        end
        for seq_id = seq_ids
            seq_name = sprintf('%s-%02d',dataset_name{1},seq_id);

            loc_test = opts.dbg_type;% 0: no test, 1: compute resize impact, 2: bbox localization error, 3: SEG with GT masks
            if loc_test == 1
                opts_debug = struct('resize_impact',1,'bb_loc_error',0,'cpn_bb_gt_seg',0,'mask_sz',25,'pad',3);
                opts_loc.seg = 1;opts_loc.tra = 0;conf_bb.scale=1;
            elseif loc_test == 2
                opts_debug = struct('resize_impact',1,'bb_loc_error',1,'cpn_bb_gt_seg',1,'mask_sz',25,'pad',3);
                opts_loc.seg = 1;opts_loc.tra = 0;
            end
            if loc_test > 0
                if train; continue; end
                gt_orig = bia.datasets.load(seq_name, {'gt'}, struct('tracked',opts_loc.tra,'segmented',opts_loc.seg,'pad',opts.use_pad));
            end
            [gt,ims] = bia.datasets.load(seq_name, {'gt','im'}, struct('scale',conf_bb.scale,'tracked',opts_loc.tra,'segmented',...
                opts_loc.seg,'pad',opts.use_pad,'version',[0 conf_bb.im_version 0], 'test', test_seqs == 0 && train == 0 ));
            T = length(ims);
            if loc_test == 1
                cpn.exp.debug(opts_debug, gt_orig, gt)
                continue;
            end

            seq_str = sprintf('%s%s',opts.str_00{length(seq_ids)},seq_name);
            if opts.verbose >= 1
                bia.print.fprintf('blue','%s-%s\n', seq_str, opts.str_train{train+1})
            end
            
            % CPN-BB
            cpn_bb_def = fullfile(conf_bb.paths.dir, sprintf('final_test%s.prototxt', conf_bb.paths.id));
            cpn_bb_weights = fullfile(conf_bb.paths.dir, sprintf('final%s', conf_bb.paths.id));
            if opts.verbose >= 2
                fprintf('CPN: Loading: %s -> %s\n',cpn_bb_def, cpn_bb_weights);
            end
            bia.caffe.clear;
            net_bb = bia.caffe.load(cpn_bb_def, cpn_bb_weights);
            if strcmp(opts.use_bb,'gt')
                bb_nms = get_gt_boxes(gt);
            elseif strcmp(opts.use_bb,'cpn')
                % conf_bb.nms = bia.utils.setfields(conf_bb.nms,'nms_overlap_thres',0.5,'after_nms_topN',2000);
                conf_bb.debug = 0;
                [bb_nms, bb_all, anchors_all] = cpn.bb.proposals(conf_bb, net_bb, ch3(ims, opts.channels));
                % fns = eval_anchors(bb_nms, bb_all, anchors_all, gt);
%                 for t=1:length(fns)
%                     gts = gt.seg.stats{gt.seg.info(:,1)==t};
%                     imshow(bia.draw.boundary([],ims{t},gts))
%                     gts = gts(fns{t});
%                     % ce = bia.convert.centroids(gts);
%                     bia.plot.centroids([], gts,'r',20)
%                     drawnow
%                 end
                fprintf('\b : ')
            end
            %figure(1)
%             imshow(ims{1},[]);bia.plot.bb([],bb_nms{1}(1:10,1:4))

            if loc_test == 2
                cpn.exp.debug(opts_debug, gt_orig, gt, bb_nms)
                continue
            end
            %if train && ~opts.save;    continue;  end

            % CPN-SEG
            if conf_bb.scale ~= conf_seg.scale
                % rescale bb_nms
                bb_nms = resize_bb(bb_nms, conf_bb.scale, conf_seg.scale);
                [gt,ims] = bia.datasets.load(seq_name, {'gt','im'}, struct('scale',conf_seg.scale,'tracked',opts_loc.tra,'segmented',...
                    opts_loc.seg,'pad',opts.use_pad,'version',[0 conf_seg.im_version 0], 'test', test_seqs == 0 && train == 0 ));                
            end
            if strcmp(opts.use_seg,'')
            elseif strcmp(opts.use_seg,'gt')
                seg_nms = get_gt_masks(gt, bb_nms);
                seg_nms = bia.struct.standardize(seg_nms, 'seg');
                if test_seqs == 0 && train == 0
                    continue
                end
            elseif strcmp(opts.use_seg,'cpn')
                cpn_seg_def = fullfile(conf_seg.paths.dir, sprintf('final_test%s.prototxt', conf_seg.paths.id));
                cpn_seg_weights = fullfile(conf_seg.paths.dir, sprintf('final%s', conf_seg.paths.id));                
                if opts.verbose >= 2
                    fprintf('SEG: Loading: %s -> %s\n',cpn_seg_def, cpn_seg_weights);
                end
                bia.caffe.clear;
                if conf_bb.im_version ~= conf_seg.im_version
                    [~,ims] = bia.datasets.load(seq_name, {'im'}, struct('scale',conf_seg.scale,'tracked',opts_loc.tra,'segmented',...
                    opts_loc.seg,'pad',opts.use_pad,'version',[0 conf_seg.im_version 0]));
                end
                net_seg = bia.caffe.load(cpn_seg_def, cpn_seg_weights);
                conf_seg.debug = 0;
                [seg_nms,~,~,bb_nms] = cpn.seg.proposals(conf_seg, net_seg, ch3(ims, opts.channels), bb_nms);
                dir_exp = conf_seg.paths.dir;
            elseif strcmp(opts.use_seg,'thresh')
                [bb_nms, seg_nms] = cpn_seg_threshold(bb_nms,ims,opts.roi_pad, keep_bb);
                dir_exp = conf_bb.paths.dir;
            end
            
            % NMS
            if strcmp(opts.nms_type,'seg')
                parfor t=1:T
                    [seg_nms{t}, pick] = bia.utils.nms(struct('iou',0.5,'use_seg',1,'score',0.001),seg_nms{t});%iou=0.2,score=0.8
                    bb_nms{t} = bb_nms{t}(pick,:);
                end
            elseif strcmp(opts.nms_type,'bb')
            end
            %figure(2)
            %mask = bia.draw.boundary([],ims{1},seg_nms{1}(1:50));
            %imshow(mask);

            if isempty(opts.use_seg)
            else
                if opts.use_pad == -1
                    seg_nms = rm_pad(seg_nms, gt.foi_border/2, gt.sz);
                    gt = bia.datasets.load(seq_name, 'gt', struct('scale',conf_seg.scale,'tracked',opts_loc.tra,'segmented',opts_loc.seg));%reload GT without any padding
                end
                eval_loc(gt, 2, seg_nms);
                if opts.save
                    seg_nms_r = bia.struct.resize(seg_nms, gt.sz_orig, gt.sz, 1);
                    save_file(opts, conf_bb, conf_seg, opts.use_seg, sprintf('%s%s%s',opts.train_str{train+1},opts.model_str,seq_str), seg_nms_r);
                end
            end
            % cpn_compute_loss(conf_bb, imdb_train, roidb_train, model);
            
            if 0;   plot_tip_fig(ims, gt, bb_nms, anchors_all, seg_nms);    end

            %% greedy detection code: todo: update later
            if ~opts.only_props
                opts = get_nms_settings(opts, train, conf_bb.paths.dir, bb_nms, gt);
                seg_greedy_s = cell(T, 1);
                bb_greedy = cell(T, 1);
                if strcmp(opts.nms_type,'seg') && ismember(opts.use_seg, {'cpn','thresh','gt'})
                    parfor t=1:T
                        [seg_greedy_s{t,1}, pick] = bia.utils.nms(struct('iou',opts.greedy_iou,'use_seg',1,'score',opts.greedy_score),seg_nms{t});%iou=0.2,score=0.8
                        bb_greedy{t} = bb_nms{t}(pick,:);
                    end
                elseif strcmp(opts.nms_type,'bb')
                    bb_greedy = do_nms(bb_nms, opts.greedy_iou, opts.greedy_score);
                    % seg_greedy_s = cpn.seg.proposals(conf_seg, net_seg, ch3(ims, opts.channels), bb_greedy);
                elseif strcmp(opts.nms_type,'ilp') && ismember(opts.use_seg, {'cpn','thresh','gt'})
                    for t=1:T
                        [seg_greedy_s{t}] = ilp_detect(seg_nms{t});
                        bb_greedy{t} = [bia.convert.bb(seg_greedy_s{t}, 's2c'), [seg_greedy_s{t}(:).Score]'];
                    end
                end
                if opts.use_pad == -1
                    seg_greedy_s = rm_pad(seg_greedy_s, gt.foi_border/2, gt.sz);
                end
                eval_loc(gt, 2, [], seg_greedy_s);
                if opts.save
                    seg_nms = bia.struct.resize(seg_nms, gt.sz_orig, gt.sz, 1);
                    seg_greedy_s = bia.struct.resize(seg_greedy_s, gt.sz_orig, gt.sz, 1);
                    save_file(opts, conf_bb, conf_seg, opts.use_seg, sprintf('%s%s%s',opts.train_str{train+1},opts.model_str,seq_str), seg_nms, seg_greedy_s);
                end
                if opts.save
                    vid_file_name = fullfile(dir_exp, sprintf('%s%s',opts.model_str,seq_str));
                    cpn_seg_video(struct('vid_file',vid_file_name, 'use_sqrt',opts.use_sqrt, 'alpha',opts.alpha,'linewidth',opts.linewidth), ims, seg_greedy_s, gt)
                end
            end
        end
    end
end
bia.caffe.clear;
end


function bb = resize_bb(bb, cpn_scale, seg_scale)
scale = seg_scale/cpn_scale;
for t=1:length(bb)% b-format: [topleft size]
    bb{t}(:,1:2) = max(1, bb{t}(:,1:2)*scale);
    bb{t}(:,3:4) = bb{t}(:,3:4)*scale;
end
end


function opts = get_nms_settings(opts, train, dir_cpn, bb_nms, gt)
nms_file = fullfile(dir_cpn, 'nms_settings.mat');
if train == 1
    time_start = tic;
    [nms_iou, thresh] = tune_greedy(bb_nms, gt, opts.verbose >= 2);
    if exist(nms_file, 'file')
        load(nms_file, 'nms_iou', 'thresh')
    end
    save(nms_file, 'nms_iou', 'thresh')
    if opts.verbose >= 2
        fprintf('Computing NMS settings (%1.2f:%1.3f) took: %1.0f sec, and saved at: %s\n', nms_iou, thresh, toc(time_start), nms_file)
    end
else
    if ~exist(nms_file, 'file')
        nms_iou = 0.1;
        thresh  = 0.9;
        if opts.verbose >= 2
            fprintf('Default NMS settings: (%1.2f:%1.3f), NMS Settings file not found: %s\n', nms_iou, thresh, nms_file)
        end
    else
        load(nms_file, 'nms_iou', 'thresh')
        if opts.verbose >= 2
            fprintf('Loaded NMS settings: (%1.2f:%1.3f) from: %s\n', nms_iou, thresh, nms_file)
        end
    end
end
opts.greedy_iou = nms_iou;
opts.greedy_score = thresh;
end


function stats = rm_pad(stats, pad, sz)
sz_new = sz - 2*pad;
parfor t=1:length(stats)
    for i=1:length(stats{t})
        tmp = new_px(stats{t}(i).PixelIdxList, pad, sz_new(t,:), sz(t,:));
        stats{t}(i)=bia.utils.setfields(stats{t}(i),'Area',tmp.Area,'BoundingBox',tmp.BoundingBox,'Centroid',tmp.Centroid,'PixelIdxList',tmp.PixelIdxList);
    end
    stats{t} = bia.struct.standardize(stats{t},'seg');
end
end


function stats = new_px(px, pad, sz_new, sz)
[r,c] = ind2sub(sz, px);
rm = r<=pad | c<=pad | r>sz(1)-pad | c>sz(2)-pad;
r(rm) = [];
c(rm) = [];
r = r-pad;
c = c-pad;
px = sub2ind(sz_new, r, c);

stats = bia.stats.pixelidxlist2stats(px, sz_new);
end


function [bb_nms, bb_greedy] = get_gt_boxes(gt)
% get GT boxes as CPN bbox proposals
if isempty(gt)
    bb_nms = cell(0);
    bb_greedy = cell(0);
    return
end
tl = gt.seg.info(gt.seg.info(:,3)==1, 1);
for t=1:gt.T
    if ismember(t, tl)
        box_tmp = bia.convert.bb(gt.seg.stats{gt.seg.info(:,1)==t}, 's2b');
    else
        box_tmp = zeros(0,4);%[10 10 50 50;100 100 50 50];
    end
    bb_nms{t,1} = [box_tmp, ones(size(box_tmp,1),1)];
end
bb_greedy = bb_nms;
end


function seg_nms = get_gt_masks(gt, bb_nms)
seg_nms = cell(gt.T, 1);
if ~isfield(gt, 'seg')
    for t = 1:gt.T
        seg_nms{t} = struct('PixelIdxList',[],'Area',0,'BoundingBox',[0.5 0.5 0 0],'Centroid',[NaN NaN],'Score',0);
    end
    return
end

parfor t = 1:gt.T
    stats = struct('PixelIdxList',{},'Area',{},'BoundingBox',{},'Centroid',{},'Score',{});
    if ismember(t, gt.seg.info(:,1)')
        mask = bia.convert.stat2im(gt.seg.stats{gt.seg.info(:,1) == t}, gt.sz(t,:));
        bb = bb_nms{t};
        sz = gt.sz;
        for k=1:size(bb,1)
            % todo: sometimes there are multiple cells within a box and actual cell for a box may have smaller area than another cell,
            % which can lead to wrong cell selection, and cause SEG score to be less than '1' for some cells.
            r  = bia.convert.bb(bb(k, 1:4), 'b2r',struct('clip',1,'sz', size(mask)));
            bw = bia.seg.largest_obj(mask(r(1):r(2), r(3):r(4)), 0);
            stats(k,1) = bia.stats.roi_stats(bw, r, sz(t,:), bb(k,5));
        end
        if isempty(bb)
            stats = struct('PixelIdxList',{},'Area',{},'BoundingBox',{},'Centroid',{},'Score',{});
        end
    else
        if isempty(bb_nms{t})
            stats = struct('PixelIdxList',{},'Area',{},'BoundingBox',{},'Centroid',{},'Score',{});
        else
            for k=1:size(bb_nms{t},1)
                stats(k,1) = struct('PixelIdxList',[],'Area',0,'BoundingBox',[0.5 0.5 0 0],'Centroid',[NaN NaN],'Score',bb_nms{t}(k,5));
            end
        end
    end
    seg_nms{t,1} = stats;
end
end

function [bb] = do_nms(bb, iou, thresh)
for t=1:length(bb)
    pick = nms([bia.convert.bb(bb{t}(:,1:4), 'b2c'), bb{t}(:,5)], iou, 0);
    bb{t} = bb{t}(pick,:);
    bb{t} = bb{t}(bb{t}(:,5)>=thresh,:);
end
end


function eval_loc(gt, bb_or_seg, seg_nms, seg_greedy)
if isempty(gt)
    return
end
if bb_or_seg == 1;    str = ['BB '];
elseif bb_or_seg == 2;    str = ['SEG'];
end

if ~isempty(seg_nms)
    [avp_tra,f1s_tra,recall,precision] = bia.metrics.ap_markers(bb_or_seg==2,seg_nms,gt,0);
    [avp_iou,f1s_iou,recall,precision] = bia.metrics.seg_ap_iou('',seg_nms,gt);
    [SEG_NMS, SEG_NMS_F] = bia.metrics.seg(struct('proposals',1),seg_nms, gt);
    fprintf('Proposals -%s: (TRA/IoU): AP:%1.3f/%1.3f, F1:%1.3f/%1.3f, SEG:(CTC:%1.3f, FULL:%1.3f)\n', str,avp_tra,avp_iou,f1s_tra,f1s_iou, SEG_NMS, SEG_NMS_F);
end

if nargin >= 4% Greedy evaluation
    [avp_tra,f1s_tra,recall,precision] = bia.metrics.ap_markers(bb_or_seg==2,seg_greedy,gt,0);
    [avp_iou,f1s_iou,recall,precision] = bia.metrics.seg_ap_iou('',seg_greedy,gt);
    [SEG_NMS, SEG_NMS_F] = bia.metrics.seg(struct('proposals',0),seg_greedy, gt);
    fprintf('Detections-%s: (TRA/IoU): AP:%1.3f/%1.3f, F1:%1.3f/%1.3f, SEG:(CTC:%1.3f, FULL:%1.3f)\n', str,avp_tra,avp_iou,f1s_tra,f1s_iou, SEG_NMS, SEG_NMS_F);
end
end


function save_file(opts, conf_cpn, conf_seg, mode, seq_name, res_nms, res_greedy)
root_export = opts.root_export;
version = '';
if ~isempty(version); version = ['-', version]; end
if isempty(mode) || ismember(mode,{'gt'})
    file_nms = fullfile(root_export, sprintf('%s-e%dm%d%s.mat',seq_name,conf_cpn.exp_id,conf_cpn.mdl_id, version));
else
    file_nms = fullfile(root_export, sprintf('%s-e%dm%d-e%dm%d%s.mat',seq_name,conf_cpn.exp_id,conf_cpn.mdl_id,conf_seg.exp_id,conf_seg.mdl_id, version));
end
stats = res_nms;
save(file_nms,    'stats')

if nargin == 7
    file_detect = strrep(file_nms, sprintf('%s.mat', version), sprintf('%s-detect.mat', version));
    stats = res_greedy;
    save(file_detect, 'stats')
end
end


function ims = ch3(ims, ch)
if ch == 3 && size(ims{1},3) == 1
    for t=1:length(ims)
        ims{t} = repmat(ims{t}, [1 1 3]);
    end
end
end


function fn = eval_anchors(bb_nms, bb_all, anchors_all, gt)
t_list = gt.seg.info(:,1)';

anchors = bia.convert.bb(anchors_all{1}, 'c2m');
anchors_sz=anchors(:,3:4);
anchor_templates = unique(anchors_sz,'rows');
[~,anchors_id_]=ismember(anchors_sz,anchor_templates,'rows');

fn = cell(max(t_list),1);
for t=t_list
    gt_bb = bia.convert.bb(gt.seg.stats{gt.seg.info(:,1)==t},'s2m');

    bb = bia.convert.bb(bb_all{t}(:,1:4), 'c2m');    
    eval_anchors_loc(gt_bb, bb, bb_all{t}(:,5),'all');
    
%     idx = find(bb_all{t}(:,5) >0.5);
%     eval_anchors_loc(gt_bb, bb(idx,:), bb_all{t}(idx,5),'>.5');
    
    bb_nms_t = bia.convert.bb(bb_nms{t}(:,1:4), 'b2m');
    fn{t} = eval_anchors_loc(gt_bb, bb_nms_t, bb_nms{t}(:,5),'nms');
    
%     eval_anchors_loc(gt_bb, anchors, zeros(size(anchors,1),1),'anc');
end
% fprintf('\n')
end


function fn = eval_anchors_loc(gt_bb, bb, scores, str)
overlaps = bia.utils.overlap_bb(gt_bb, bb);
[o, idx] = max(overlaps, [], 2);
scores = scores(idx,1);
fn = find(o<0.5);
% figure
% histogram(scores)
% fprintf('%s:: t:%3d, #GT: %5d, #Props: %6d, IoU<:: 0.5->%d, 0.7->%d\n', str, 0, size(gt_bb,1), size(bb,1), sum(o<0.5), sum(o<0.7))
end
