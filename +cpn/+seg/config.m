function conf = config(varargin)

ip = inputParser;
ip = cpn.utils.config(ip);

% training
ip.addParameter('roi_old', false,    @islogical);% identifies the stage
ip.addParameter('stage', 'seg',    @isstr);% identifies the stage
ip.addParameter('batch_size',      256,            @isscalar);% Minibatch size
ip.addParameter('fg_fraction',     1,       @isscalar);% Fraction of minibatch that is foreground labeled (class > 0)
ip.addParameter('fg_thresh',       0.5,            @isscalar);% Overlap threshold for a ROI to be considered foreground (if >= fg_thresh)
ip.addParameter('bg_thresh_hi',    0.3,            @isscalar);% Overlap threshold for a ROI to be considered background (class = 0 if overlap in [bg_thresh_lo, bg_thresh_hi))
ip.addParameter('bg_thresh_lo',    0.1,            @isscalar);
ip.addParameter('bbox_thresh',     0.5,            @isscalar);% Vaild training sample (IoU > bbox_thresh) for bounding box regresion
ip.addParameter('mode', 1, @isscalar);% 1: predict only SEG, 2: predict SEG + refine BBOX

ip.addParameter('pos_criteria', 2, @isscalar);%1(use markers), 2(use iou), 3(use markers+iou), 5(% of pixel occupied by main mask >0.3)
ip.addParameter('ignore_invalid', 1, @isscalar);
ip.addParameter('use_weights', false, @islogical);%use weights to prioritize boundary pixels
% ip.addParameter('pos_criteria_seg', 0, @isscalar);% use seg instead of bbox

% test + train
ip.addParameter('refine_bbox', false,        @islogical);% 1: refine bbox in the 2nd stage
ip.addParameter('use_padding', true,         @islogical);% 
ip.addParameter('mask_type', 2, @isscalar);% 1(all FG pixels), 2(only largest cell)
ip.addParameter('mask_sz', 25, @isscalar);% feature map size after ROI-poling and output mask size
ip.addParameter('roi_pad', 3, @isscalar);
ip.addParameter('roi_fixed', 0, @isscalar);% >0 (use fixed size (==roi_fix) roi)

% solver
ip.addParameter('momentum', 0.99, @isscalar);
ip.addParameter('base_lr', 0.0001, @isscalar);

ip.parse(varargin{:});
conf = ip.Results;


conf = cpn.io.set_paths(conf, 'seg');

% seg model
opts_mdl = struct('id',conf.mdl_id,'type','seg','channels',conf.channels,'mask_sz',conf.mask_sz,'init','gaussian');
[~, train_net, test_net] = cpn.io.save_model(opts_mdl);

path_train_def = fullfile(conf.paths.dir, sprintf('train%s.prototxt', conf.paths.id));
path_solver = fullfile(conf.paths.dir, sprintf('solver%s.prototxt', conf.paths.id));

bia.caffe.save_net(train_net, path_train_def)
bia.caffe.save_net(test_net, fullfile(conf.paths.dir, sprintf('test%s.prototxt', conf.paths.id)))

solver = struct('net', path_train_def,'base_lr',conf.base_lr,'lr_policy','step','momentum',conf.momentum,'stepsize',conf.stepsize,...
    'max_iter',conf.max_iter,'weight_decay',conf.weight_decay,'gamma',0.1,...
    'display',1000,'snapshot',0);% solver setting
bia.caffe.save_solver(solver, path_solver)

end