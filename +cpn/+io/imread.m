function im = imread(im_path, ~, opts_deform)
% cpn.io.imread(file_name)
% cpn.io.imread(file_name, version)
% cpn.io.imread(file_name, version, deform_opts)
persistent ims

if nargin == 0
    ims = struct();
    return
end


if nargin == 3
    deform = opts_deform.map_deform;
    frames = opts_deform.frames;
    pad = opts_deform.pad;
    fields = fieldnames(frames);
    for i=1:length(fields)
        for t=frames.(fields{i}).t
            im_orig = ims.(fields{i}){t};
            sz = size(im_orig);
            im = padarray(im_orig, [pad pad], 'symmetric');
            im = imwarp(im, deform(1:sz(1)+2*pad,1:sz(2)+2*pad,:),'linear');%'cubic'
            im = im(pad+1:end-pad, pad+1:end-pad, :);
            szx = size(im);
            assert(isequal(sz, szx))
            ims.(fields{i}){t} = im;
        end
    end
    return
end

[dataset, t, scale, gt_version, im_version, rot, flip, channels, split] = cpn_decode_name(im_path);
d = strrep(dataset, '-', '_');
if ~isfield(ims, sprintf('%s', d))
    % fprintf('##################   Loading images: %s   ##################\n', dataset)
    [~,ims.(d)]=bia.datasets.load(dataset,{'im'}, struct('scale',scale,'version',[gt_version, im_version, 0]));
end
im = ims.(d){t};

if channels == 3 && size(im, 3)~=3; im = repmat(im, [1 1 3]);   end
im = flip(im);
im = rot(im);

if isstruct(split) && split.num_splits > 1; im = cpn_crop_im(im, [], [], split);    end

end
