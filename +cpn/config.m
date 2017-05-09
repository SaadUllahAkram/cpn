function conf = config(varargin)
% config file for cpn experiments

ip = inputParser;

ip.addParameter('verbose',  1,  @isscalar);% 1: only the results, 2: add other important info, 3: print everything
ip.addParameter('use_bb',  'cpn',  @isstr);% 'cpn'(boxes from cpn), 'gt' (gt boxes)
ip.addParameter('nms_type',  'seg',  @isstr);% 'seg'(use seg), 'bb'(use boxes), 'ilp'(use ilp)
ip.addParameter('use_seg',  'cpn',  @isstr);% '': only bb from cpn, 'gt': GT masks, 'cpn': cpn seg, 'thresh': threshold
ip.addParameter('process',  'whole',  @isstr);% 'whole': do whole seq, 'seg': only seg, 'tra': only tracked, 'seg_fully': only fully seg
ip.addParameter('dataset',  'Fluo-N2DL-HeLa',  @isstr);% 'Fluo-N2DL-HeLa','Fluo-N2DH-GOWT1','PhC-C2DH-U373','PhC-C2DL-PSC','Hist-BM','PhC-HeLa-Ox'


ip.addParameter('use_pad',  0,  @isscalar);% -1: use foi border as padding, 0: dont use padding
ip.addParameter('seg_pad',  3,  @isscalar);% 
ip.addParameter('dbg_type', 0,  @isscalar);%
ip.addParameter('eval_train',  [0 1],  @ismatrix);% 0: testing run, 1: training run
ip.addParameter('train_seq',   [0],  @ismatrix);% which sequences to use for training:: 1 or 2: Trained using "01"/"02" seq, 0: trained using both seqs


ip.addParameter('only_props',  true,  @islogical);% only eval/save props
ip.addParameter('dont_train',  true,  @islogical);% 1: dont train even if the model does not exist
ip.addParameter('do_eval',  true,  @islogical);%
ip.addParameter('skip_seg',  false,  @islogical);%
ip.addParameter('do_val',  true,  @islogical);%
ip.addParameter('save',  true,  @islogical);% 1: save proposals, 0: dont save proposals
ip.addParameter('root_export',  '',  @isstr);% path where proposals will be saved
ip.addParameter('root_train',  '',  @isstr);% path where trained models and intermediary files will be saved


ip.parse(varargin{:});
conf = ip.Results;

end