function [train_progress, val_progress] = plot_loss(train_progress, val_progress, train_results, val_results, ax_loss, leg)
% Plots accuracy in 1st subplot and loss in 2nd plot
% 
% Inputs:
%     ax_loss, leg, opts.do_val
% 

line_width  = 1;
colors      = {'r', 'g', 'b', 'k'};
y_labels    = {'log(accuracy)', 'log(loss)'};

if isempty(val_results)
    do_val = 0;
else
    do_val = 1;
end

if isempty(train_progress)
    train_progress{1} = cell(1, length(leg{1}));
    train_progress{2} = cell(1, length(leg{2}));
end
if isempty(val_progress)
    val_progress{1} = cell(1, length(leg{1}));
    val_progress{2} = cell(1, length(leg{2}));
end

m = repmat([Inf -Inf Inf -Inf], 2, 1);
legends = leg;
verif   = {'accu', 'loss'};% to confirm that struct is not unexpected
for j=1:2% 1->accuracy, 2->loss
    cla(ax_loss(j));
    hold(ax_loss(j), 'on')
    
    n = length(legends{j});
    if n == 0
        continue;
    end
    for i = 1:n
        try
            assert(~isempty(strfind(leg{j}{i}, verif{j})))
        catch
            warning('Unexpected value in legend: %s\n', leg{j}{i})
        end
        
        if j == 1
            train_progress{j}{i} = [train_progress{j}{i}, 1 - mean(train_results.(leg{j}{i}).data)];
        else
            train_progress{j}{i} = [train_progress{j}{i}, mean(train_results.(leg{j}{i}).data)];
        end
        m = plot_shortcut(ax_loss(j), train_progress{j}{i}, j, ['-', colors{i}], line_width, m);
        legends{j}{(2*i-1)*(do_val==1) + i*(do_val==0)}   = strrep(leg{j}{i}, '_', '\_');
        
        if do_val
            if j == 1
                val_progress{j}{i} = [val_progress{j}{i}, 1 - mean(val_results.(leg{j}{i}).data)];
            else
                val_progress{j}{i} = [val_progress{j}{i}, mean(val_results.(leg{j}{i}).data)];
            end
            m = plot_shortcut(ax_loss(j), val_progress{j}{i}, j, ['--', colors{i}], line_width, m);
            legends{j}{(2*i)*(do_val==1)} = [strrep(leg{j}{i}, '_', '\_'), '-Val'];
        end
    end
    legend(ax_loss(j), rep(legends{j}), 'FontSize',6,'Location','southwest');
    ylabel(ax_loss(j), y_labels{j})
    xlabel(ax_loss(j), 'Iteration')
end
drawnow
end


function l = rep(l)
for i=1:length(l)
    l{i} = strrep(l{i}, 'loss\_','l\_');
    l{i} = strrep(l{i}, 'accuracy\_','a\_');
end
end


function m = plot_shortcut(ax, y, type, ls, lw, m)
eps = 10^-6;
y = log(eps+y);% +eps->to prevent log from going to Inf, 
x = 1:length(y);
plot(ax, x, y, ls, 'linewidth', lw)
offset = 0.2;
if sum(isnan(y))
    y = 0;
end
axis(ax, [min(x)-offset max(x)+offset min(y)-offset max(y)+offset])
m(type,1) = min(m(type,1), min(x));
m(type,2) = max(m(type,2), max(x));
m(type,3) = min(m(type,3), min(y));
m(type,4) = max(m(type,4), max(y));
axis(ax, [m(type,1)-offset m(type,2)+offset m(type,3)-offset m(type,4)+offset])
grid(ax,'on')

end
