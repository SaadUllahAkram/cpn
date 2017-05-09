function conf_cpn = setup_anchors(conf_cpn, dataset)
% sets anchor sizes and generates feature map size (# of anchors for an image size)
% Also sets CPN setting within conf
%
% Inputs:
%     net_model.{model.stride}
%     model
%     conf_cpn
% Outputs:
%     conf_cpn
% 

anchors = [];
file_maps = conf_cpn.paths.out_sz;
szs = [get_sizes(dataset); [conf_cpn.max_dim conf_cpn.max_dim]];
% conf_cpn.redo_map = 1;
while (~isequal(conf_cpn.anchors, anchors))
    if ~conf_cpn.redo_map && exist(file_maps, 'file')
        load(file_maps, 'output_map', 'anchors', 'anchors_offset')
        if ~isequal(conf_cpn.anchors, anchors)
            warning('Anchors miss-match: Re-computing CPN output size map\n')
            delete(file_maps)
        end
        if ~exist('anchors_offset','var')%old
            bia.print.fprintf('*red', 'Loading wrong "anchors_offset"\n')
            anchors_offset = 0;
        end
    else
        fprintf('Computing CPN output size map took: ')
        t_start = tic;
        anchors = conf_cpn.anchors;
        [output_map, anchors_offset] = cpn.bb.feat_map_size(conf_cpn, fullfile(conf_cpn.paths.dir, sprintf('test%s.prototxt', conf_cpn.paths.id)), szs);
        save(file_maps, 'output_map', 'anchors', 'anchors_offset')
        fprintf('%1.1f sec\n', toc(t_start))
    end
end
conf_cpn = bia.utils.setfields(conf_cpn,'output_map', output_map, 'anchors_offset', anchors_offset);

if nargin == 2
    cpn.bb.eval_anchors(dataset.roidb_train, conf_cpn.anchors);
end
end


function szs = get_sizes(dataset)
szs = cell2mat(arrayfun(@(x) x.sz, dataset.roidb_train, 'UniformOutput', false));
szs = unique(szs,'rows');
if ~isempty(dataset.roidb_val)
    szs_val = cell2mat(arrayfun(@(x) x.sz, dataset.roidb_val, 'UniformOutput', false));
    szs_val = unique(szs_val,'rows');
    szs = [szs; szs_val];
end
szs = szs(:, 1:2);
end