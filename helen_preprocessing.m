clc
clear all
close all

%%
%Loading annotation
for i = 1:2330 %2330 files
    filename = ['/Users/brysont/Downloads/annotation/' num2str(i) '.txt']; %Input annotation files
    fid = fopen(filename,'r');

    n = 195; %195 lines in each file
    for lines = 1:n
        a{i,lines} = fgetl(fid);
        b{i,lines} = str2num(char(strsplit(a{i,lines},',')));
    end
    
    %cropping scope
    c{i}(3) = b{i,42}(1)-b{i,2}(1); %length x3
    c{i}(4) = (b{i,22}(2)-b{i,2}(2))*2; %width x4
    c{i}(2) = b{i,2}(2)*2-b{i,22}(2); %Yleftup x2
    c{i}(1) = b{i,2}(1); %Xleftup x1
    d{i} = scope_zoom(c{i},1.5);
    
    %Looking for new coordinate, e{} is new cropped coordinate
    for t = 2:n
        e{i,t}(1) = b{i,t}(1) - d{i}(1);
        e{i,t}(2) = b{i,t}(2) - d{i}(2);
    end
    
    h{i,1}(1) = (e{i,147}(1) + e{i,155}(1))/2; %Left eye
    h{i,1}(2) = (e{i,147}(2) + e{i,155}(2))/2;
    h{i,2}(1) = (e{i,116}(1) + e{i,127}(1))/2; %Right eye
    h{i,2}(2) = (e{i,116}(2) + e{i,127}(2))/2;
    h{i,3}(1) = (e{i,58}(1) + e{i,45}(1))/2; %Nose
    h{i,3}(2) = (e{i,58}(2) + e{i,45}(2))/2;
    h{i,4}(1) = e{i,60}(1); %Left side of mouth
    h{i,4}(2) = e{i,60}(2);
    h{i,5}(1) = e{i,75}(1); %Right side of mouth
    h{i,5}(2) = e{i,75}(2);
    
    x1{i} = [h{i,1}(1), h{i,2}(1), h{i,3}(1), h{i,4}(1), h{i,5}(1); h{i,1}(2), h{i,2}(2), h{i,3}(2), h{i,4}(2), h{i,5}(2)];
    
end

x2 = zeros(2,5);
%%
for i = 1:2330    
    %First cropping
    filename1 = ['/Users/brysont/Desktop/Helen/' a{i,1} '.jpg']; %Loading filename1
    %filename2 = ['/Users/brysont/Desktop/Helen/out1/' a{i,1} '.png']; %Output filename2
    I = imread(filename1); %Loading images
    size_of_cropped_img1 = d{i}(3); %Length
    size_of_cropped_img2 = d{i}(4); %Width
    x_cent = b{i,22}(1); %Xmidpoint
    y_cent = b{i,2}(2); %Ymidpoint
    xmin = x_cent-size_of_cropped_img1/2; 
    ymin = y_cent-size_of_cropped_img2/2; 
    I2 = imcrop(I,[xmin ymin size_of_cropped_img1 size_of_cropped_img2]); %After cropping, I2
    %imwrite(I2,filename2); %Output
    mysize=size(I2);     
    height = mysize(1); % Line num
    width = mysize(2);  % Column num
    I3 = imresize(I2,[(height/width)*400 400],'nearest'); %Downsample
    x3{i} = (x1{i}/width)*400;
    x2 = x2 + x3{i}; %Sum
end

x4 = x2/2330; %Expectation

%%
%Rigid transform and last cropping
for i = 1:2330
    filename2 = ['/Users/brysont/Desktop/Helen/out1/' a{i,1} '.png']; %Output filename2
    I2 = imread(filename2); %Loading images
    I3 = imresize(I2,[(height/width)*400 400],'nearest'); %Downsample
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
    size_of_cropped_img1 = mysize(2)/1.2; %Length
    size_of_cropped_img2 = mysize(1)/1.2; %Width
    x_cent = mysize(2)/2; %Xmidpoint
    y_cent = mysize(1)/2; %Ymidpoint
    xmin = x_cent-size_of_cropped_img1/2; 
    ymin = y_cent-size_of_cropped_img2/2; 
    I5 = imcrop(I4,[xmin ymin size_of_cropped_img1 size_of_cropped_img2]); %After cropping, I5
    I6 = imresize(I5,[256 256],'nearest'); %Size transforming 256*256
    imwrite(I6,filename4); %Output
    
end

fclose(fid);

%%
function X_out = scope_zoom(X_in,factor) %1.5 times  X_in(x,y) 
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
function [scale,R,T]=dh_match(x1,x2,display_flag)    %x1(x,y) x2(x,y)    

if display_flag == true
    figure(1);
    plot(x1(:,1),x1(:,2),'r*',x2(:,1),x2(:,2),'b*');
    set(gca,'Ydir','reverse'); 
    title('Before alignment');

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
function X_output = coordinate_transformation(X_input,R,T,scale)   
    X = X_input;
    X(:,1)=X(:,1)*scale;
    X(:,2)=X(:,2)*scale;
    X = bsxfun(@plus, R*X', T);
    X_output=X';
end

%%
function new_I = convert_image_from_I(image_I,scale,R,T)  %%Whole image transform
I = image_I;
%figure
%imshow(I);
mysize=size(I);     
height = mysize(1); % Lines
width = mysize(2);  % Columns
channel = mysize(3);
channel = numel(mysize); %Pixels

for c=1:1:channel
    for i=1:1:width
        for j=1:1:height
            a = [i j]; 
            b = coordinate_transformation(a,R,T,scale);
            b = round(b);
            if( (b(1)>0) && (b(1)<width) && (b(2)> 0) && (b(2)< height)) 
                %new_I(b(1),b(2),c) = I(j,i,c);
                new_I(b(2),b(1),:) = I(j,i,:); 
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
function [scale,R,T] = optimal_trans(x1,x2)  

error_min=10000; %Bigger is needed
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
