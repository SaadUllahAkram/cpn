function eval_anchors(roidb, anchors)
% Match anchors with GT boxes to figure out which anchors are good and which are bad (not needed)

anchors = anchors_center(anchors);
N = length(roidb);
oo = [];
n_gt_loc = zeros(1,N);
for i=1:N
    [~, bbox_tl_br] = cpn.io.gtread(roidb(i).image_id);
    o = match(bbox_tl_br, anchors);
    oo = [oo; [o, i*ones(size(o,1), 1)]];% [overlap, anchor_id, gt_id, image_id]
    n_gt_loc(i) = size(bbox_tl_br,1);
    % im = cpn.io.imread(roidb(i).image_id);
    % imshow(im)
    % bia.plot.bb([], bia.convert.bb(bbox_tl_br, 'c2m'),'r')
end
n_gt = sum(n_gt_loc);

cc5=0;cc7=0;
for t=1:max(oo(:,4))
   a = oo(oo(:,4)==t,:);
   for k=1:max(a(:,3))
       b = a(a(:,3)==k,:);
       if max(b(:,1)) > 0.7
           cc7 = cc7+1;
       elseif max(b(:,1)) > 0.5
           cc5 = cc5+1;
       end
   end
end

for i=1:size(anchors,1)
    a=oo(oo(:,2)==i,:);
    counts(:,i) = [sum(a(:,1)> 0.7); sum(a(:,1)> 0.5)];
    fprintf('Anchors #: %d, IoU>0.7: %6d, IoU>0.5: %6d\n', i, counts(1,i), counts(2,i))
%     figure(i)
%     hist(a(:,1))
end
counts(:, end+1) = sum(counts,2);
fprintf('#GT: %d : #Anchors:: IoU>0.7: %d, IoU>0.5: %d\n', n_gt, counts(1,end), counts(2,end));
fprintf('#GT: %d : #Cells:: IoU>0.7: %d, IoU>0.5: %d :: #UnMatched: %d\n', n_gt, cc7, cc5, n_gt-(cc5+cc7));
fprintf('Anchors (%% Matched):\n');
disp(100*counts./n_gt)
end


function oo = match(gt, anchors)
cents = [sum(gt(:,[1 3]), 2)/2, sum(gt(:,[2 4]), 2)/2];
m = size(gt,1);
n = size(anchors,1);
oo = [];
for i=1:m
    a = anchors;
    a(:,[1,3]) = a(:,[1,3]) + repmat(cents(i,1), n, 2);
    a(:,[2,4]) = a(:,[2,4]) + repmat(cents(i,2), n, 2);
    a = bia.convert.bb(a,'c2m');
    b = bia.convert.bb(gt(i,:),'c2m');
    o = bia.utils.overlap_bb(a,b);
    oo = [oo; [o, [1:n]', i*ones(n,1)]];
end
end


function anchors = anchors_center(anchors)
x = 0-anchors(:,1);
anchors(:,[1, 3]) = anchors(:,[1, 3]) + repmat(x, 1, 2);
y = 0-anchors(:,2);
anchors(:,[2, 4]) = anchors(:,[2, 4]) + repmat(y, 1, 2);

anchors(:, [1,3]) = anchors(:, [1,3]) - repmat(anchors(:,3)/2,1,2);
anchors(:, [2,4]) = anchors(:, [2,4]) - repmat(anchors(:,4)/2,1,2);
end
