% Code for visualizing Wij
% written by Seung-Chan
% https://www.tensorflow.org/get_started/mnist/beginners

clc; clear; close all;
%A = load('out/w_0.txt');
l = dir('./out/*.txt');

for k=1:size({l.name}, 2)
    fn = sprintf('out/%s', l(k).name)
    A = load(fn);

    f = figure;
    for i=1:size(A, 2)
       Ai = A(:,i);
       idxn = find(Ai <0);
       idxp = find(Ai>=0);

       AAn = zeros(size(A, 1),1);
       AAp = zeros(size(A, 1),1);

       AAp(idxp) = Ai(idxp); 
       AAn(idxn) = Ai(idxn); 

       AAp = reshape(AAp, 28, 28);
       AAn = reshape(AAn, 28, 28);

       Ip = mat2gray(AAp);
       In = mat2gray(abs(AAn));

       img = zeros(28,28,3);
       img(:,:, 1) = In';
       img(:,:, 2) = Ip';
       img(:,:, 3) = Ip';

       subplot(2,5,i);
       image(img);
       ttl = sprintf('#%d', i-1);
       title(ttl);
       axis equal;
       xlim([1,28]);
       ylim([1,28]);

    end

    % Create textbox
    underlineLocations = find(fn == '_');
    dotLocations = find(fn == '.');
    nIt = str2double(fn(underlineLocations(1)+1:dotLocations(1)));
    
    ttl1 = sprintf('Iteration #%d', nIt);
    annotation(f,'textbox',...
        [0.45 0.92 0.15 0.05],...
        'String',{ttl1},...
        'LineStyle','none',...
        'FitBoxToText','off');
    fnout = sprintf('out/iter_%d.png', nIt);
    saveas(f, fnout);
end
