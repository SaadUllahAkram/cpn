function train(opts, conf_bb, conf_seg)
cpn.io.imread()
cpn.io.gtread()
warning('off','MATLAB:dispatcher:UnresolvedFunctionHandle')% to handle changed imdb_fun name & path
close all
test_seqs = 2*(opts.train_seq==1) + 1*(opts.train_seq==2);

bia.print.fprintf('*blue','#### pid:%d EXPERIMENT: [%d-%d] [%d-%d] ####\n', feature('getpid'), conf_bb.exp_id, conf_bb.mdl_id, conf_seg.exp_id, conf_seg.mdl_id)

% TRAIN BB
cpn_bb_config = fullfile(conf_bb.paths.dir, sprintf('config%s.mat', conf_bb.paths.id));
cpn_bb_final = fullfile(conf_bb.paths.dir,  sprintf('final%s', conf_bb.paths.id));
if exist(cpn_bb_config,'file') && exist(cpn_bb_final,'file')% load the model and training settings
    load(cpn_bb_config)
elseif opts.dont_train % do nothing
else
    imdb = cpn.io.get_imdb(conf_bb);
    ensure_new(conf_bb.paths);
    [dataset_train, dataset_val, dataset_debug, opts.do_val] = cpn.io.progress_data('',conf_bb, [], opts.do_val, imdb);% get the val/progress data
    conf_bb = cpn.bb.setup_anchors(conf_bb, imdb);% generate anchors
    cpn.bb.train(conf_bb, dataset_train, dataset_val, dataset_debug);
end

% TRAIN SEG
cpn_seg_config = fullfile(conf_seg.paths.dir, sprintf('config%s.mat', conf_seg.paths.id));
cpn_seg_final = fullfile(conf_seg.paths.dir,  sprintf('final%s', conf_seg.paths.id));
if opts.skip_seg
elseif exist(cpn_seg_config,'file') && exist(cpn_seg_final,'file')
    load(cpn_seg_config)
elseif opts.dont_train % do nothing
else
    ensure_new(conf_seg.paths);
    imdb = cpn.io.get_imdb(conf_seg);
    [dataset_train, dataset_val, dataset_debug, opts.do_val, ~, conf_seg] = cpn.io.progress_data('',conf_bb,conf_seg,opts.do_val,imdb);% get the val/progress data
    cpn.seg.train(conf_seg, dataset_train, dataset_val, dataset_debug);
end


if opts.do_eval
    cpn.test(opts, conf_bb, conf_seg, opts.dataset, test_seqs);
end
bia.caffe.clear;

end


function ensure_new(paths)
iter5k = fullfile(paths.dir, sprintf('iter_%d%s', 5000, paths.id));
if exist(iter5k, 'file')
    error('Did previous training not finish? : %s', iter5k)
end
end