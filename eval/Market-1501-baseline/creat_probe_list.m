clc
clear

prob_dir = 'dataset/query/';
gallery_dir = 'dataset/bounding_box_test/';
prob_files = dir([prob_dir '*.jpg']);
gallery_files = dir([gallery_dir '*.jpg']);
f_prob=fopen('prob.txt','w+');
f_gallery=fopen('gallery.txt','w+');
for n = 1:length(prob_files)
    n
    img_name = prob_files(n).name;
    if strcmp(img_name(1), '-') % junk images
        ID = -1;
    else
        ID = str2num(img_name(1:4));
    end
            
    
    str=['query/' img_name ' ' num2str(ID)];
   
    fprintf(f_prob,'%s\n',str);
        
        
        
        
    
end
for n = 1:length(gallery_files)
    n
    img_name = gallery_files(n).name;        
    if strcmp(img_name(1), '-') % junk images
        ID = -1;
    else
        ID = str2num(img_name(1:4));
    end
    str=['bounding_box_test/' img_name ' ' num2str(ID)];
   
    fprintf(f_gallery,'%s\n',str);    
end
fclose(f_prob);
fclose(f_gallery);
    
