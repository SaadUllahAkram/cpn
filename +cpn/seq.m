function seq_num = seq(seq_num, expand)
% seq_num == 0 -> both are training data
% expand: 0: returns same seq or expands '0' -> [1 2]
% expand: 1: swaps sequences 1->2, and vice versa, also expands '0' -> [1 2]

if expand == 1% get training sequences
    if seq_num == 0
        seq_num = 1:2;
    else
        seq_num = rem(seq_num,2)+1;% get the training seq number
    end
elseif expand == 0
    if seq_num == 0% test
        seq_num = 1:2;
    end
end

end
