clear
close all
tic
%% Nature scence test
Img = double(imread('cameraman.tif'));
Img = imresize(Img,[500,500]);
Img = (Img - min(min(Img))) / (max(max(Img))-min(min(Img)));


Pattern_length = 10;
Img_Z = Z_scanning(Img, Pattern_length);
figure
subplot(1,2,1)
imagesc(Img);colormap('gray');axis image off;colorbar();title('GT');
subplot(1,2,2)
imagesc(Img_Z);colormap('gray');axis image off;colorbar();title('Z scanned');
sgtitle('Example of nature scence');
%% Straight line test
Img2 = zeros(500);
Img2(10:10:end,:) = ones(50,500);
Img_Z2 = Z_scanning(Img2, Pattern_length);
figure
subplot(1,2,1)
imagesc(Img2);colormap('gray');axis image off;colorbar();title('GT');
subplot(1,2,2)
imagesc(Img_Z2);colormap('gray');axis image off;colorbar();title('Z scanned');
sgtitle('Straight line test');

toc
function output = Z_scanning(input, length)
[d1,d2] = size(input);
output = zeros(d1,d2);
scale_intepolation = d2/length;
input_padd = zeros(d1+length,d2);
input_padd(1:d1,1:d2) = input;
for i = 1 : round(d1/length/2)
    for j = 1 : d2
        col_partial = imresize(input_padd((i-1)*length*2+1:(2*i+1)*length,j),...
            [scale_intepolation*length*3,1],'nearest');
        Upper = conv2(col_partial(1+j-1:d2+j-1,1),...
            1/scale_intepolation*ones(scale_intepolation,1),'valid'); 
        output((2*i-2)*length+1:(2*i-1)*length,j) = Upper(1:scale_intepolation:end,1);
        Lower = conv2(col_partial(1+2*d2-(j-1):3*d2-(j-1),1),...
            1/scale_intepolation*ones(scale_intepolation,1),'valid') ;
        output((2*i-1)*length+1:i*length*2,j) = Lower(1:scale_intepolation:end,1);
    end
end
end