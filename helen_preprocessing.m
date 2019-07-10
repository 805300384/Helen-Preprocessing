clc
clear all
close all

%%
%读取标注
for i = 1:2330 %2330个文件
    filename = ['/Users/brysont/Downloads/annotation/' num2str(i) '.txt']; %输入annotation文件
    fid = fopen(filename,'r');

    n = 195; %每个文件195行
    for lines = 1:n
        a{i,lines} = fgetl(fid);
        b{i,lines} = str2num(char(strsplit(a{i,lines},',')));
    end
    
    %裁剪范围
    c{i}(3) = b{i,42}(1)-b{i,2}(1); %长 x3
    c{i}(4) = (b{i,22}(2)-b{i,2}(2))*2; %宽 x4
    c{i}(2) = b{i,2}(2)*2-b{i,22}(2); %左上角纵坐标 x2
    c{i}(1) = b{i,2}(1); %左上角横坐标 x1
    d{i} = scope_zoom(c{i},1.5);
    
    %寻找新坐标,e为裁剪后的坐标点
    for t = 2:n
        e{i,t}(1) = b{i,t}(1) - d{i}(1);
        e{i,t}(2) = b{i,t}(2) - d{i}(2);
    end
    
    h{i,1}(1) = (e{i,147}(1) + e{i,155}(1))/2; %左眼
    h{i,1}(2) = (e{i,147}(2) + e{i,155}(2))/2;
    h{i,2}(1) = (e{i,116}(1) + e{i,127}(1))/2; %右眼
    h{i,2}(2) = (e{i,116}(2) + e{i,127}(2))/2;
    h{i,3}(1) = (e{i,58}(1) + e{i,45}(1))/2; %鼻子
    h{i,3}(2) = (e{i,58}(2) + e{i,45}(2))/2;
    h{i,4}(1) = e{i,60}(1); %左嘴角
    h{i,4}(2) = e{i,60}(2);
    h{i,5}(1) = e{i,75}(1); %右嘴角
    h{i,5}(2) = e{i,75}(2);
    
    x1{i} = [h{i,1}(1), h{i,2}(1), h{i,3}(1), h{i,4}(1), h{i,5}(1); h{i,1}(2), h{i,2}(2), h{i,3}(2), h{i,4}(2), h{i,5}(2)];
    
end

x2 = zeros(2,5);
%%
for i = 1:2330    
    %第一次裁剪
    filename1 = ['/Users/brysont/Desktop/Helen/' a{i,1} '.jpg']; %读取filename1
    %filename2 = ['/Users/brysont/Desktop/Helen/out1/' a{i,1} '.png']; %输出filename2
    I = imread(filename1); %读取图片
    size_of_cropped_img1 = d{i}(3); %长
    size_of_cropped_img2 = d{i}(4); %宽
    x_cent = b{i,22}(1); %中点横坐标
    y_cent = b{i,2}(2); %中点纵坐标
    xmin = x_cent-size_of_cropped_img1/2; %需要对x_cent提前赋值,边长size_of_cropped_img赋值
    ymin = y_cent-size_of_cropped_img2/2; %需要对y_cent提前赋值
    I2 = imcrop(I,[xmin ymin size_of_cropped_img1 size_of_cropped_img2]); %第一次裁剪后的图保存为I2
    %imwrite(I2,filename2); %输出
    mysize=size(I2);     % 发现：
    height = mysize(1); % 行数
    width = mysize(2);  % 列数
    I3 = imresize(I2,[(height/width)*400 400],'nearest'); %降采样
    x3{i} = (x1{i}/width)*400;
    x2 = x2 + x3{i}; %求和
end

x4 = x2/2330; %求均值

%%
%刚体变换，并进行最后一次裁剪
for i = 1:2330
    filename2 = ['/Users/brysont/Desktop/Helen/out1/' a{i,1} '.png']; %输出filename2
    I2 = imread(filename2); %读取图片
    I3 = imresize(I2,[(height/width)*400 400],'nearest'); %降采样
    filename3 = ['/Users/brysont/Desktop/Helen/out2/' a{i,1} '.png'];
    [scale,R,T]=dh_match(x3{i}',x4',false);
    x1_out = coordinate_transformation(x3{i}',R,T,scale);
    I4 = convert_image_from_I(I3,scale,R,T);
    %figure;
    %imshow(I4);
    %imwrite(I4, filename3);
    filename4 = ['/Users/brysont/Desktop/Helen/out3/' a{i,1} '.png'];
    %I4 = imread(filename3);
    mysize=size(I4);
    size_of_cropped_img1 = mysize(2)/1.2; %长
    size_of_cropped_img2 = mysize(1)/1.2; %宽
    x_cent = mysize(2)/2; %中点横坐标
    y_cent = mysize(1)/2; %中点纵坐标
    xmin = x_cent-size_of_cropped_img1/2; %需要对x_cent提前赋值,边长size_of_cropped_img赋值
    ymin = y_cent-size_of_cropped_img2/2; %需要对y_cent提前赋值
    I5 = imcrop(I4,[xmin ymin size_of_cropped_img1 size_of_cropped_img2]); %第二次裁剪后的图保存为I5
    I6 = imresize(I5,[256 256],'nearest'); %变换尺寸为256*256
    imwrite(I6,filename4); %输出
    
end

fclose(fid);

%%
function X_out = scope_zoom(X_in,factor) % 可以取1.5倍  X_in(x,y) 1为左上角横坐标，2为左上角纵坐标，3为长，4为宽
    mid_x = X_in(1)+X_in(3)/2;
    mid_y = X_in(2)+X_in(4)/2;
    X_out = [0 0 0 0];
    X_out(1) = mid_x - X_in(3)/2*factor;
    X_out(2) = mid_y - X_in(4)/2*factor;
    X_out(3) = X_in(3)*factor;
    X_out(4) = X_in(4)*factor;
    X_out = round(X_out);
end

%%
function [scale,R,T]=dh_match(x1,x2,display_flag)    %x1(x,y) x2(x,y)    %%查看变换结果

if display_flag == true
    figure(1);
    plot(x1(:,1),x1(:,2),'r*',x2(:,1),x2(:,2),'b*');
    %需要把坐标轴Y方向反向,因为图像是左上角为原点，而绘图坐标轴是左下角为原点
    set(gca,'Ydir','reverse'); %Y轴反向
    title('Before alignment');

    %%对两对关键点求取映射公式  X方向和Y方向 尺度因子应该一样

    %[scale_x,scale_y,R,T] = optimal_trans(x1,x2)
    x1_train = x1;
    x2_train = x2;

    [scale,R,T] = optimal_trans(x1_train,x2_train);

    x1(:,1)=x1(:,1)*scale;
    x1(:,2)=x1(:,2)*scale;

    %[R_test, T_test] = rigidTransform2D(x1, x2);  
    %x1 = bsxfun(@plus, R_test*x1', T_test);

    x1 = bsxfun(@plus, R*x1', T);
    x1=x1';
    %error=sum((x1(:,1)-x2(:,1)).^2+(x1(:,2)-x2(:,2)).^2)

    figure(2);
    plot(x1(:,1),x1(:,2),'r*',x2(:,1),x2(:,2),'b*');
    set(gca,'Ydir','reverse');
    title('After alignment');
else
    x1_train = x1;
    x2_train = x2;
    [scale,R,T] = optimal_trans(x1_train,x2_train);
end

end

%%
function X_output = coordinate_transformation(X_input,R,T,scale)   % X_input(x,y) %按照 变换参数 进行坐标变换
    X = X_input;
    X(:,1)=X(:,1)*scale;
    X(:,2)=X(:,2)*scale;
    X = bsxfun(@plus, R*X', T);
    X_output=X';
end

%%
function new_I = convert_image_from_I(image_I,scale,R,T)  %%对整个图片进行变换
I = image_I;
%figure
%imshow(I);
mysize=size(I);     % 发现： 红外图也是三通道的
height = mysize(1); % 行数
width = mysize(2);  % 列数
channel = mysize(3);
channel = numel(mysize); %图像通道数

for c=1:1:channel
    for i=1:1:width
        for j=1:1:height
            a = [i j]; % 注意matlab 图像矩阵的x,y顺序  , 此处是a(x,y)
            b = coordinate_transformation(a,R,T,scale);
            b = round(b);
            if( (b(1)>0) && (b(1)<width) && (b(2)> 0) && (b(2)< height)) %变换未超出原本的图像范围 
                %new_I(b(1),b(2),c) = I(j,i,c);
                new_I(b(2),b(1),:) = I(j,i,:); % 注意matlab 图像矩阵的x,y顺序 !! 这一块之前就写错了
            else 
                b(1);
                b(2);
            end
        end
    end
end
%figure
%imshow(new_I);
end

%%
function [scale,R,T] = optimal_trans(x1,x2)  %%求取变换参数  x1(x,y) x2(x,y)

error_min=10000; %要设的大一些，因为出现了 error没有小于error_min的情况
%scale_x = 0;
%scale_y = 0;
scale = 0;
x1_origin = x1;

for factor=0.1:0.0005:2
    %for factor_y=0.5:0.01:1.5
        %if factor>0.64
            %factor
        %end
        x1(:,1)=x1_origin(:,1)*factor;
        x1(:,2)=x1_origin(:,2)*factor;
        [R_train, T_train] = rigidTransform2D(x1, x2);  
        x1 = bsxfun(@plus, R_train*x1', T_train);
        x1=x1';

        error=sum((x1(:,1)-x2(:,1)).^2+(x1(:,2)-x2(:,2)).^2);
        
        if error < error_min
           error_min = error;
           %error_min
           scale = factor;
           R = R_train;
           T = T_train;           
        end
        
    %end
end
end

%%
function [R, t] = rigidTransform2D(p, q)

n = cast(size(p, 1), 'like', p);
m = cast(size(q, 1), 'like', q);

% Find data centroid and deviations from centroid
pmean = sum(p,1)/n;
p2 = bsxfun(@minus, p, pmean);

qmean = sum(q,1)/m;
q2 = bsxfun(@minus, q, qmean);

% Covariance matrix
C = p2'*q2;

[U,~,V] = svd(C);

% Handle the reflection case
R = V*diag([1 sign(det(U*V'))])*U';

% Compute the translation
t = qmean' - R*pmean';

end