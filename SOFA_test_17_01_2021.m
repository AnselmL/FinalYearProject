clear all;
addpath(genpath('API_MO-master'));

SOFAstart;
base_path = 'E:\FYP\MATLAB\eBrIRD';
file_name = 'anechoic.sofa';
full_path = strcat(base_path,'\',file_name);
hrtf = SOFAload(full_path);

BRIR_anechoic = hrtf.Data.IR;

BRIR_anechoic_dim = size(BRIR_anechoic);

%{

file_name = 'kitchen.sofa';
full_path = strcat(base_path,'\',file_name);
hrtf = SOFAload(full_path);

BRIR_kitchen = hrtf.Data.IR;

BRIR_kitchen_dim = size(BRIR_kitchen);

file_name = 'restaurant.sofa';
full_path = strcat(base_path,'\',file_name);
hrtf = SOFAload(full_path);

BRIR_restaurant = hrtf.Data.IR;

BRIR_restaurant_dim = size(BRIR_restaurant);

%}

%save(strcat(base_path,'\BRIR_data'),'BRIR_anechoic','BRIR_anechoic_dim','BRIR_kitchen','BRIR_kitchen_dim','BRIR_restaurant','BRIR_restaurant_dim');
