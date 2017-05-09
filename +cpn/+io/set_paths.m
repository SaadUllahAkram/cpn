function conf = set_paths(conf, stage)
% set paths of network related files
% 
% Inputs:
%     conf:
%     stage: 'bb' or 'seg'
%     mode: 'ohem' or ''
% Outputs:
%     conf:
% 
if conf.ohem
    mode = '_ohem';
else
    mode = '';
end
stage = ['_', stage];
root_cpn = conf.root_train;

dataset = conf.dataset;
train_seq = conf.train_seq;

root = fullfile(root_cpn, sprintf('%04d_m%d_%s-%02d%s', conf.exp_id, conf.mdl_id, dataset, train_seq, stage));
com_str = sprintf('_cpn%s%s', stage, mode);
bia.save.mkdir(root);
conf.paths = struct('dir',root,...
    'id',com_str,...
    'imdb',root_cpn,...
    'init_net_file','');

if strcmp(stage, '_bb')
    conf.paths.out_sz = fullfile(root, sprintf('cpn_bb_out_sizes.mat'));
end
path_final = fullfile(root, sprintf('final%s', conf.paths.id));
if ~isempty(mode)
   conf.paths.init_net_file = path_final;
end
if exist(path_final, 'file')
    assert(exist(fullfile(conf.paths.dir, sprintf('final_test%s.prototxt', conf.paths.id)), 'file') == 2)
end

end