clc
clear

train_dir = 'dataset/bounding_box_train/';
val_dir = 'dataset/bounding_box_test/';
train_files = dir([train_dir '*.jpg']);
val_files = dir([val_dir '*.jpg']);
f_train=fopen('train.txt','w+');
f_val=fopen('val.txt','w+');
for n = 1:length(train_files)
    n
    img_name = train_files(n).name;        
    ID = str2num(img_name(1:4))-1;
    str=['bounding_box_train/' img_name ' ' num2str(ID)];
    m=rand();
    if  m<=0.8
        fprintf(f_train,'%s\n',str);
   
    else
        fprintf(f_val,'%s\n',str);
    end   
end

fclose(f_train);
fclose(f_val);
    
