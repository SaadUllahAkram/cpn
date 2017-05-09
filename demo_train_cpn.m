% demo for training cpn and performing cell tracking
bia.caffe.clear;

dataset = 'Fluo-N2DL-HeLa';
exp_id = 1;

bia.compile()
cpn.build()
% bia.caffe.activate('cpn',-1)

paths = get_paths();
root_train = paths.save.cpn;
root_export = paths.save.cpn_res;

% import and preprocess data
bia.datasets.import.ctc(dataset);
bia.datasets.import.ctc_fluo_hela_aug();% hela augmented data using watershed

% train cpn models
for train_seq = 1:2
    opts = cpn.config('dataset', dataset, 'train_seq', train_seq, 'dont_train', false, 'root_train', root_train, 'root_export', root_export);
    conf_bb = cpn.bb.config(dataset, train_seq, 1, 1,'gt_version',1,'im_version',0,'scale',1, 'root_train', root_train);
    conf_seg = cpn.seg.config(dataset, train_seq, 2, 1,'gt_version',0,'im_version',0,'scale',1, 'root_train', root_train);
    cpn.train(opts, conf_bb, conf_seg);
end
bia.caffe.clear;