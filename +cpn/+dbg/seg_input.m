function seg_input(conf, in)
im = uint8(in{1})';
rois = squeeze(in{2})';
rois = rois(:,2:5)+1;%1-based indexing

masks = squeeze(in{3})';
weights = squeeze(in{4})';

[~,hax1]=bia.plot.fig('a');
[~,hax2]=bia.plot.fig('b');
figure(11)
drawnow
imshow(im,'parent',hax1)
bb = bia.convert.bb(rois, 'c2m');
bia.plot.bb(hax1, bb)

[stats, o] = rec(masks, rois, size(im));
drawnow
imshow(bia.draw.boundary([], im, stats),'parent',hax2)

end


function [stats, o] = rec(masks, rois, im_sz)
n = size(rois,1);
o = cell(n,1);
w = sqrt(size(masks,2));

rect_crop         = round(rois(:,[2 4 1 3]));% rois=[xmin ymin xmax ymax] -> rect_crop=[ymin ymax xmin xmax]
rect_crop(:,[1,3])= max(1, rect_crop(:,[1,3]));
rect_crop(:,[2,4])= min(repmat([im_sz(1), im_sz(2)], size(rect_crop, 1), 1), rect_crop(:,[2,4]));
for i=1:n
    px = reshape(masks(i,:), w, w)';
    r   = rect_crop(i, :);
    bsz = [r(2)-r(1)+1, r(4)-r(3)+1];
    o{i}  = imresize(px, bsz, 'nearest');
    
    stats(i,1) = cpn_roi_stats(o{i}, r, im_sz, 0, 0);
%     px  = imresize(px, bsz, 'bicubic');
end
end



function stats = cpn_roi_stats(mask, roi, sz, pad, score, feats)
% rois: []
% mask: binary mask
% sz: size of full image
% stats:

idx     = find(mask);
[r,c]   = ind2sub(size(mask), idx);
r       = double(r+roi(1)-1-pad);
c       = double(c+roi(3)-1-pad);
idx_rm  = r < 1 | c < 1 | r > sz(1) | c > sz(2);
r(idx_rm) = [];
c(idx_rm) = [];

stats.PixelIdxList = sub2ind(sz, r, c);
stats.Area = length(stats.PixelIdxList);

% code snippets copied from "regionprops"
num_dims = numel(sz);
list = [c, r];
if (isempty(list))
    stats.BoundingBox = [0.5*ones(1,num_dims) zeros(1,num_dims)];
else
    min_corner = min(list,[],1) - 0.5;
    max_corner = max(list,[],1) + 0.5;
    stats.BoundingBox = [min_corner (max_corner - min_corner)];
end
stats.Centroid = mean([r,c],1);
stats.Score = score;
if nargin >= 6
    stats.Features = feats;
end
end
