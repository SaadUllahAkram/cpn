function imdb_name = get_imdb_name(dataset_name, net_type, imdb_id)
aug_str = {'', '_ws'};
expansion_str = {'_orig','','_deform', '_exact'};
use_expansion = 3;

% bia.print.fprintf('red', 'Training using data from: %s-%02d\n', dataset_name, seq)
if imdb_id == 0% get imdb containing orig data as it is
    augment = 0;
    use_expansion = 0;
    max_sz = Inf;
else
%     if ismember(dataset_name, {'Fluo-N2DH-GOWT1','Fluo-N2DL-HeLa'})
    if ismember(dataset_name, {'Fluo-N2DL-HeLa'})
        if ismember(net_type, {'cpn'})
            warning('Using version 1 of GT for training')
            augment = 1;
        else
            augment = 0;
        end
        max_sz = 500;
%     elseif ismember(dataset_name, {'PhC-C2DH-U373','PhC-C2DL-PSC','Hist-BM','PhC-HeLa-Ox'})
    elseif ismember(dataset_name, {'Fluo-N2DH-GOWT1','PhC-C2DH-U373','PhC-C2DL-PSC','Hist-BM','PhC-HeLa-Ox'})
        augment = 0;
        max_sz = 500;
    end
end
imdb_name = @(x) sprintf('%s-%d%s%s_sz%d_%s.mat', dataset_name, x, aug_str{augment+1}, expansion_str{use_expansion+1}, max_sz, net_type);
end