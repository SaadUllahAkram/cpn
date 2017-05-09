function [iou_invalid, invalid, gt_rois] = foiread(image_path, rois)

% Gets IoU of bboxes with invalid regions
% 

[invalid,gt_rois] = cpn.io.gtread(image_path);
integral_im     =  integralImage(invalid);
sz              = size(invalid);

r           = round(rois(:,[2, 4, 1, 3]));% (l, t, r, b) to (t b l r)
r(:,[1,3])  = min( repmat(sz(1:2), size(r,1), 1),  max(1, r(:,[1,3])) );
r(:,[2,4])  = min( repmat(sz(1:2), size(r,1), 1), r(:,[2,4]));

h           = r(:,2)-r(:,1);
w           = r(:,4)-r(:,3);
area        = h.*w;

tl  = sub2ind(sz+1, r(:,2)+1-h, r(:,4)+1-w);
tr  = sub2ind(sz+1, r(:,2)+1-h, r(:,4)+1);
bl  = sub2ind(sz+1, r(:,2)+1, r(:,4)+1-w);
br  = sub2ind(sz+1, r(:,2)+1, r(:,4)+1);
area_invalid    = integral_im(br) + integral_im(tl) - integral_im(tr) - integral_im(bl);
iou_invalid     = area_invalid./area;
end
