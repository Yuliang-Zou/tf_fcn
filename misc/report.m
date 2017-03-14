run('/Users/chengao/Desktop/tf_fcn-master/data/VOCdevkit/VOCcode/VOCinit.m')
[accuracies,avacc,conf,rawcounts] = Evaluation(VOCopts,'32rgb');

figure(1)
hBar = bar(accuracies / 100);
Xt = 1 : length(accuracies);
Xl = [0 22];
set(gca, 'XTick', Xt, 'XLim', Xl,'FontSize', 15);

label = [' background';
         '  aeroplane';
         '    bicycle';
         '       bird'; 
         '       boat';
         '     bottle';
         '        bus';
         '        car';
         '        cat';
         '      chair';
         '        cow';
         'diningtable';
         '        dog';
         '      horse';
         '  motorbike';
         '     person';
         'pottedplant';
         '      sheep';
         '       sofa';
         '      train';
         '  tvmonitor'];

ax = axis;   
axis(axis);   
Yl = ax(3:4);  % Y-axis limits

t = text(Xt,Yl(1) * ones(1,length(Xt)),label(1:21,:),'FontSize',20);
set(t,'HorizontalAlignment','right','VerticalAlignment','top', ...
      'Rotation',45);

set(gca,'XTickLabel','')



