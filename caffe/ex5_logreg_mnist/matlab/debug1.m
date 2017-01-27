% for displaying parameters being updated (iteration : 30,000)
% Seung-Chan 

clc; clear; close all;
A = load('../outbig/out_idx_0.txt');

figure;
for i=1:size(A, 2)
    plot(A(:, i)); hold on;
end

a = sprintf('Iteration : %d, parameters : %d', size(A, 1), size(A, 2));
title(a);
xlabel('Iteration');