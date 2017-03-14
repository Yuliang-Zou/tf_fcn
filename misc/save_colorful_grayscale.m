% Description:  save grayscale segmentation to RGB
%
% Author:       Chen Gao
%               chengao@umich.edu
%
% Date:         March 7, 2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function save_colorful_grayscale(in_directory,out_directory)

cmap     = VOClabelcolormap(256);
contents = dir([in_directory,'*.png']);


for idx = 1:numel(contents)
    
    idx
    filename = contents(idx).name;
    img_gray = imread([in_directory filename]);   
    imwrite(img_gray,cmap,[out_directory filename]);
    
end


function cmap = VOClabelcolormap(N)

if nargin==0
    N = 256;
end
cmap = zeros(N,3);
for i=1:N
    id = i-1; r=0;g=0;b=0;
    for j=0:7
        r = bitor(r, bitshift(bitget(id,1),7 - j));
        g = bitor(g, bitshift(bitget(id,2),7 - j));
        b = bitor(b, bitshift(bitget(id,3),7 - j));
        id = bitshift(id,-3);
    end
    cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;