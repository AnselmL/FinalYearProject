Invalid window length. WINDOWLENGTH must be in the range [2,size(x,1)], where x is the audio input.
The default window length depends on the specified sample rate: round(fs*0.03).clear all;
addpath(genpath('E:\FYP\MATLAB\MBSTOI'));
addpath(genpath('E:\FYP\MATLAB\submodules'));
addpath(genpath('E:\FYP\MATLAB\other_dependencies'));
addpath(genpath('E:\FYP\MATLAB\RIR-Generator-master'));
addpath(genpath('E:\FYP\MATLAB\ANF-Generator-master'));


%{
[x, fs_x] = audioread('anechoic_speech.wav');
y = rand(size(x,1),1);

[z,p,fs_o] = v_addnoise(x,fs_x,10,'dxopEkn',y,fs_x);
%}

TIMIT_wav_dirs = load('TIMIT_wav_dirs').TIMIT_wav_dirs;

%TIMIT_wav_dirs_test = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_test;
%TIMIT_wav_dirs_train = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_train;
%TIMIT_wav_dirs_val = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_val;

measured_BRIR_wav_dirs = load('measured_BRIR_wav_dirs').BRIR_wav_dirs;


base_dir = 'E:\FYP\MATLAB';
RIR_dir = 'E:\FYP\Room_IR_Database';
RIR_file_ext = '.wav';

out_base_dir = 'D:\FYP\ML_data_dir\directional\measured\10-fold_2021_04_10';

%out_base_unnorm_dir = 'E:\FYP\ML_data_dir\simulated\unnorm\1-fold';

noise_wav_dirs = load('TUT_wav_dirs').TUT_wav_dirs;


                


%az_target_sm_rm = -180:12.5:170;

%az_noise = [90];
%el_noise = [0];

%az_el_target = [0,0]; %this can also be matrix if multiple target azimuths and elevations are desired
%az_el_noise = [90,0]; %this can be matrix with column 1 for azimuth angle and column 2 for elevation angle
%row matrix of all desired target radii, specified in meters

%radii_noise = [1,3,5]; %row matrix of all desired noise radii, specified in 
%{
noise_name_array = {'NOISEX-92','babble';
                    'NOISEX-92','buccaneer1';
                    'NOISEX-92','buccaneer2';
                    'NOISEX-92','destroyerengine';
                    'NOISEX-92','destroyerops';
                    'NOISEX-92','f16';
                    'NOISEX-92','factory1';
                    'NOISEX-92','factory2';
                    'NOISEX-92','hfchannel';
                    'NOISEX-92','leopard';
                    'NOISEX-92','m109';
                    'NOISEX-92','machinegun';
                    'NOISEX-92','pink';
                    'NOISEX-92','volvo';
                    'NOISEX-92','white';
                    'ETSI-Binaural','Airport_Binaural';
                    'ETSI-Binaural','Cafeteria_bin';
                    'ETSI-Binaural','Callcenter1_bin';
                    'ETSI-Binaural','Callcenter2_bin';
                    'ETSI-Binaural','Conference1_bin';
                    'ETSI-Binaural','Conference2_bin';
                    'ETSI-Binaural','Conference3_bin';
                    'ETSI-Binaural','Crossroadnoise_bin';
                    'ETSI-Binaural','FullSizeCar_80_bin';
                    'ETSI-Binaural','FullSizeCar_100_bin';
                    'ETSI-Binaural','FullSizeCar_130_bin';
                    'ETSI-Binaural','Inside_Bus_bin';
                    'ETSI-Binaural','Inside_Train_bin';
                    'ETSI-Binaural','MidSizeCar_80_bin';
                    'ETSI-Binaural','MidSizeCar_100_bin';
                    'ETSI-Binaural','MidSizeCar_130_bin';
                    'ETSI-Binaural','Pub_bin';
                    'ETSI-Binaural','Roadnoise_bin';
                    'ETSI-Binaural','RockMusic01m48k_binaural';
                    'ETSI-Binaural','SalesCounter_bin';
                    'ETSI-Binaural','TrainStation_bin';
                    'ETSI-HomeLike','All_DUT1_AC1';
                    'ETSI-HomeLike','All_DUT1_AC2';
                    'ETSI-HomeLike','All_DUT2_AC1';
                    'ETSI-HomeLike','All_DUT2_AC1';
                    'ETSI-HomeLike','All_DUT3_AC1';
                    'ETSI-HomeLike','All_DUT3_AC2';
                    'RSG-10','SIGNAL001-20kHz';
                    'RSG-10','SIGNAL002-20kHz';
                    'RSG-10','SIGNAL003-20kHz';
                    'RSG-10','SIGNAL004-20kHz';
                    'RSG-10','SIGNAL024-20kHz';
                    };

%}
mic_pos_Kayser_ie = [-0.0700,0,0;
                    0.0700,0,0];
                
%{             
mic_pos_Kayser_bte = [0.0820,-0.0136,0.0374;
                     -0.0820,-0.0136,0.0374;
                     0.0820,-0.0285,0.0353;
                     -0.0820,-0.0285,0.0353];
                    %}
mic_pos_Kayser_bte = [-0.08200,-0.02105,0.03635;
                      0.08200,-0.02105,0.03635];
                  
mic_pos_eBrIRD_bte = [-0.0800,-0.0200,0.0250;
                     0.0800,-0.0200,0.0250];
                 
mic_pos_AIR = [-0.0850,0,0;
               0.0850,0,0];

mic_pos_eBrIRD_ie = [-0.0350,0,0;
                    0.0350,0,0];
%{
mic_pos_eBrIRD_bte = [0.0800,-0.0140,0.0250;
                     -0.0800,-0.0140,0.0250;
                     0.0800,-0.0260,0.0250;
                     -0.0800,-0.0260,0.0250];
                 %}


mic_pos_BKwHA_ie = [-0.0700,0,0;
            0.0700,0,0];
%{
mic_pos_BKwHA_bte = [0.0800,-0.0136,  0.0374;
            -0.0800,-0.0136,0.0374;
            0.0800,-0.0285,0.0327;
            -0.0800,-0.0285,0.0327];
            %}
        
mic_pos_BKwHA_bte = [-0.0800,-0.02105,  0.03555;
            0.0800,-0.02105,0.03555];
        
mic_pos_KEMAR_ie = [-0.0700,0,0;
            0.0700,0,0];

        
mic_pos_DADEC_ie = [-0.0600,0,0;
            0.0600,0,0];
        
mic_pos_Head_ie = [-0.0600,0,0;
            0.0600,0,0];

TIMIT_wav_pool = [];
TIMIT_cat_pool = [];
%TIMIT_train_pool = [];
%TIMIT_val_pool = [];
%TIMIT_test_pool = [];
measured_BRIR_pool = [];
noise_index_pool = [];
SNR_diff_index_pool = [];
SNR_direc_index_pool = [];
az_index_pool = [];
el_index_pool = [];
alpha_add_index_pool = [];
alpha_index_pool = [];
x_room_dim_index_pool = [];
y_room_dim_index_pool = [];
z_room_dim_index_pool = [];
radius_ratio_index_pool = [];
measured_vs_simulated_pool = [];
anechoic_mic_position_pool = [];
Kayser_eBrIRD_IR_type_pool = [];

direc_pool = [];
direc_type_pool = [];


TIMIT_train_file_count = 1;
TIMIT_test_file_count = 1;

%TIMIT_train_pool = 1:size(TIMIT_wav_dirs_train,2);
%TIMIT_val_pool = 1:size(TIMIT_wav_dirs_val,2);
%TIMIT_test_pool = 1:size(TIMIT_wav_dirs_test,2);

%mic_pos = gen_mic_pos_Kayser_2020_12_17();
nb_folds = 2;

%dimensions = [2.5,2.5,3; 4,4,3; 6,6,3; 7.5,7.5,3;10,10,4; 12.5,12.5,4; 15,15,4;10,20,4]; % room dimensions specified with [x,y,z] coordinates

dimension_x = 2:20;
dimension_y = 2:20;
dimension_z = 2:10;

alpha_plasterboard_ceilling = 0.0983;
alpha_wood_ceilling = 0.147;

alpha_plaster = 0.0283;
alpha_plywood = 0.2;

alpha_carpet = 0.182;
alpha_wood_f = 0.05;

alpha_add = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7];

alpha_array = [1, 1, 1, 1, 1, 1;
               1, 1, 1, 1, 1, 1;
               1, 1, 1, 1, 1, 1;
         alpha_plaster, alpha_plaster, alpha_plaster, alpha_plaster,alpha_carpet, alpha_plasterboard_ceilling;
         alpha_plaster, alpha_plaster, alpha_plaster, alpha_plaster,alpha_carpet, alpha_wood_ceilling;
         alpha_plaster, alpha_plaster, alpha_plaster, alpha_plaster,alpha_wood_f, alpha_plasterboard_ceilling;
         alpha_plaster, alpha_plaster, alpha_plaster, alpha_plaster,alpha_wood_f, alpha_wood_ceilling;
         alpha_plywood, alpha_plywood, alpha_plywood, alpha_plywood,alpha_carpet, alpha_plasterboard_ceilling;
         alpha_plywood, alpha_plywood, alpha_plywood, alpha_plywood,alpha_carpet, alpha_wood_ceilling;
         alpha_plywood, alpha_plywood, alpha_plywood, alpha_plywood,alpha_wood_f, alpha_plasterboard_ceilling;
         alpha_plywood, alpha_plywood, alpha_plywood, alpha_plywood,alpha_wood_f, alpha_wood_ceilling];
     
%alpha = [0.6,0.4,0.2]; %absorption coefficients
rt60 = [];
az_target = -180:10:170;
el_target = [-10,0,10];
radii_target_ratio = 0.3:0.1:0.7;
SNR_diff = -15:3:30;
SNR_direc = -6:3:15;
direc_type = [0,1]; %0 -> noise source, 1 -> speech
direc_on = [0,1]; %0 -> no directional interfer, 1 -> directional interfer

direc_mes = 1;
for fold = 1:nb_folds
    if isempty(TIMIT_wav_pool)
        TIMIT_wav_pool = 1:size(TIMIT_wav_dirs,1);
    end
    %{
    if isempty(measured_BRIR_pool)
        measured_BRIR_pool = 1:size(measured_BRIR_wav_dirs,1);
    end
    %}
    while(~isempty(TIMIT_wav_pool))
        clear BRIR
        clear BRIR_direc

    %measured_BRIR_pool = 1:size(measured_BRIR_wav_dirs,1);
    %TIMIT_cat = ["train","val","test"];

    %this is where the while(TIMIT_train_pool or TIMIT_val_pool or
    %TIMIT_test_pool)
    %{
    if isempty(TIMIT_cat_pool)
        TIMIT_cat_pool = 1:size(TIMIT_cat,2);
    end
    %}
        if isempty(measured_BRIR_pool)
            measured_BRIR_pool = 1:size(measured_BRIR_wav_dirs,1);
        end
        if isempty(noise_index_pool)
            %noise_index_pool = 1:size(noise_name_array,1);
            noise_index_pool = 1:size(noise_wav_dirs,1);
        end
        if isempty(SNR_diff_index_pool)    
            SNR_diff_index_pool = 1:length(SNR_diff);
        end
        if isempty(SNR_direc_index_pool)    
            SNR_direc_index_pool = 1:length(SNR_direc);
        end
        if isempty(az_index_pool)
            az_index_pool = 1:length(az_target);
        end
        if isempty(el_index_pool)
            el_index_pool = 1:length(el_target);
        end
        if isempty(alpha_add_index_pool)
            alpha_add_index_pool = 1:length(alpha_add);
        end
        if isempty(alpha_index_pool)
            alpha_index_pool = 1:size(alpha_array,1);
        end
        if isempty(x_room_dim_index_pool)
            x_room_dim_index_pool = 1:length(dimension_x);
        end
        if isempty(y_room_dim_index_pool)
            y_room_dim_index_pool = 1:length(dimension_y);
        end
        if isempty(z_room_dim_index_pool)
            z_room_dim_index_pool = 1:length(dimension_z);
        end
        if isempty(radius_ratio_index_pool)
            radius_ratio_index_pool = 1:length(radii_target_ratio);
        end
        if isempty(measured_vs_simulated_pool)
            measured_vs_simulated_pool = 1:2;
        end
        if isempty(anechoic_mic_position_pool)
            anechoic_mic_position_pool = 1:6;
        end
        if isempty(Kayser_eBrIRD_IR_type_pool)
            Kayser_eBrIRD_IR_type_pool = 1:2;
        end
        if isempty(direc_pool)
            direc_pool = 1:2;
        end
        if isempty(direc_type_pool)
            direc_type_pool = 1:2;
        end

        %TIMIT_wav_index = nearest(rand*size(TIMIT_wav_pool,1)+1);
        TIMIT_wav_index = TIMIT_wav_pool(randperm(size(TIMIT_wav_pool,2),1));
        TIMIT_wav_pool=TIMIT_wav_pool(TIMIT_wav_pool~=TIMIT_wav_index);
        anechoic_speech_dir = TIMIT_wav_dirs{TIMIT_wav_index,1};
        TRAIN_strfind = strfind(anechoic_speech_dir,'TRAIN');
        TEST_strfind = strfind(anechoic_speech_dir,'TEST');
        if ~isempty(TRAIN_strfind)
            out_data_subdir = 'TRAIN';
        elseif ~isempty(TEST_strfind)
            out_data_subdir = 'TEST';
        else
            error('Neither train nor test were found in the anechoic speech directory used')
        end
        [anechoic_speech, fs_speech] = audioread(anechoic_speech_dir);
        %{
        TIMIT_cat_index = nearest(rand*size(noise_name_array,1)); %1-3 train,test,val
        TIMIT_index = nearest(rand*size(TIMIT_wav_dirs,1));
        prev_used_TIMIT_index = [];
        SNR_index = nearest(rand*size(SNR,2)); %SNR index goes from 1-21
        prev_used_SNR_index = []; 
        alpha_index = nearest(rand*size(alpha,2)); %1-3
        prev_used_alpha_index = [];
        room_dim_index = nearest(rand*size(dimensions,1)); %1-12 and <4 are sm_room
        prev_used_room_dim_index = [];
        radius_index = nearest(rand*size(radii_target,1)); %1-2
        prev_used_radius_index = [];
        %}
        %fetching TIMIT anechoic sentence

        %{
        TIMIT_cat_index = nearest(rand*size(TIMIT_cat_pool,2)+1);
        while(TIMIT_cat_index == 1 && isempty(TIMIT_train_pool) || TIMIT_cat_index == 2 && isempty(TIMIT_val_pool) || TIMIT_cat_index == 3 && isempty(TIMIT_test_pool))
            TIMIT_cat_index = nearest(rand*size(TIMIT_cat_pool,2)+1);
        end
        TIMIT_cat_pool=TIMIT_cat_pool(TIMIT_cat_pool~=TIMIT_cat_index);

        if TIMIT_cat_index == 1 
            TIMIT_train_index = nearest(rand*size(TIMIT_train_pool,2)+1);
            TIMIT_train_pool=TIMIT_train_pool(TIMIT_train_pool~=TIMIT_train_index);
            anechoic_speech_dir = TIMIT_wav_dirs_train{1,TIMIT_train_index};
            [anechoic_speech, fs_speech] = audioread(anechoic_speech_dir);
        elseif TIMIT_cat_index == 2 
            TIMIT_val_index = nearest(rand*size(TIMIT_val_pool,2)+1);
            TIMIT_val_pool=TIMIT_val_pool(TIMIT_val_pool~=TIMIT_val_index);
            anechoic_speech_dir = TIMIT_wav_dirs_val{1,TIMIT_val_index};
            [anechoic_speech, fs_speech] = audioread(anechoic_speech_dir);
        else
            TIMIT_test_index = nearest(rand*size(TIMIT_test_pool,2)+1);
            TIMIT_test_pool=TIMIT_test_pool(TIMIT_test_pool~=TIMIT_test_index);
            anechoic_speech_dir = TIMIT_wav_dirs_test{1,TIMIT_test_index};
            [anechoic_speech, fs_speech] = audioread(anechoic_speech_dir);
        end
        %}
        if ~isempty(measured_BRIR_pool)
            measured_vs_simulated_index = measured_vs_simulated_pool(randperm(size(measured_vs_simulated_pool,2),1));
            measured_vs_simulated_pool=measured_vs_simulated_pool(measured_vs_simulated_pool~=measured_vs_simulated_index);
        else
            measured_vs_simulated_index = 2;
        end
        measured_vs_simulated_index = 1;
        %fetching speech impulse response
        if  measured_vs_simulated_index == 1
            measured_BRIR_index = measured_BRIR_pool(randperm(size(measured_BRIR_pool,2),1));
            %measured_BRIR_index = nearest(rand*size(measured_BRIR_pool,2)+1);
            measured_BRIR_pool=measured_BRIR_pool(measured_BRIR_pool~=measured_BRIR_index);
            BRIR_dir = measured_BRIR_wav_dirs{measured_BRIR_index,1};
            [HRTF, fs_brir] = audioread(BRIR_dir);
            if fs_brir ~= fs_speech
                fs_brir = resample(fs_brir,fs_speech,fs_brir);
            end
            %checking type of measured BRIR
            Kayser_strfind = strfind(BRIR_dir,'Kayser');
            AIR_strfind = strfind(BRIR_dir,'Aachen');
            eBrIRD_strfind = strfind(BRIR_dir,'eBrIRD');
            if ~isempty(Kayser_strfind)
                Kayser_office_I_strfind = strfind(BRIR_dir,'office_I');
                if isempty(Kayser_office_I_strfind)    
                    Kayser_eBrIRD_IR_type_index = Kayser_eBrIRD_IR_type_pool(randperm(size(Kayser_eBrIRD_IR_type_pool,2),1));
                    Kayser_eBrIRD_IR_type_pool = Kayser_eBrIRD_IR_type_pool(Kayser_eBrIRD_IR_type_pool~= Kayser_eBrIRD_IR_type_index);
                else
                    Kayser_eBrIRD_IR_type_index = 3;
                end
                if Kayser_eBrIRD_IR_type_index == 1
                    %BTE
                    BRIR(:,1) = (HRTF(:,3) + HRTF(:,7))/2;
                    BRIR(:,2) = (HRTF(:,4) + HRTF(:,8))/2;

                    mic_position = mic_pos_Kayser_bte;
                elseif Kayser_eBrIRD_IR_type_index == 2
                    %IR
                    BRIR(:,1) = HRTF(:,1);
                    BRIR(:,2) = HRTF(:,2);

                    mic_position = mic_pos_Kayser_ie;
                elseif Kayser_eBrIRD_IR_type_index == 3
                    BRIR(:,1) = (HRTF(:,1) + HRTF(:,5))/2;
                    BRIR(:,2) = (HRTF(:,2) + HRTF(:,6))/2;
                else
                    error('something went wrong')
                end
                %{
                %data augmentation
                speech_type = nearest(rand(1));
                if speech_type == 0
                    %simply TIMIT
                elseif speech_type == 1
                    %convolve TIMIT with Kayser anechoic frontal 0.8m impulse
                    %response
                end
                %}

            elseif ~isempty(AIR_strfind)
                mic_position = mic_pos_AIR;
                BRIR = HRTF;
            elseif ~isempty(eBrIRD_strfind)
                Kayser_eBrIRD_IR_type_index = Kayser_eBrIRD_IR_type_pool(randperm(size(Kayser_eBrIRD_IR_type_pool,2),1));
                Kayser_eBrIRD_IR_type_pool = Kayser_eBrIRD_IR_type_pool(Kayser_eBrIRD_IR_type_pool~= Kayser_eBrIRD_IR_type_index);
                if Kayser_eBrIRD_IR_type_index == 1
                    %BTE
                    HRTF = HRTF(:,3:6);
                    BRIR(:,1) = (HRTF(:,1) + HRTF(:,3))/2; %left brir
                    BRIR(:,2) = (HRTF(:,2) + HRTF(:,4))/2; %right brir
                    mic_position = mic_pos_eBrIRD_bte;
                elseif Kayser_eBrIRD_IR_type_index == 2
                    %IE
                    BRIR = HRTF(:,1:2);
                    mic_position = mic_pos_eBrIRD_ie;
                else
                    error('something went wrong')
                end
                %{
                %data augmentation
                speech_type = nearest(rand(1));
                if speech_type == 0
                    %simply TIMIT
                elseif speech_type == 1
                    %convolve TIMIT with Kayser anechoic frontal 0.8m impulse
                    %response
                end
                %}
            else
                error('something went wrong')
            end


        elseif measured_vs_simulated_index == 2 
            %simulated
            %mic_position_type = nearest(rand(6));
            mic_position_type = anechoic_mic_position_pool(randperm(size(anechoic_mic_position_pool,2),1));
            anechoic_mic_position_pool= anechoic_mic_position_pool(anechoic_mic_position_pool~= mic_position_type);
            if mic_position_type == 0
                mic_position = mic_pos_Kayser_bte;%Kayser BTE
            elseif mic_position_type == 1
                mic_position = mic_pos_Kayser_ie;
            elseif mic_position_type == 2
                mic_position = mic_pos_BKwHA_bte;
            elseif mic_position_type == 3
                mic_position = mic_pos_BKwHA_ie;
            elseif mic_position_type == 4
                mic_position = mic_pos_DADEC_ie;
            elseif mic_position_type == 5
                mic_position = mic_pos_Head_ie;
            elseif mic_position_type == 6
                mic_position = mic_pos_KEMAR_ie;
            else
                error('something went wrong')
            end
            %{
            speech_type = nearest(rand(1));

            if speech_type == 0

            elseif speech_type == 1

                if mic_position_type == 0
                    %convolve with Kayser BTE anechoic 0.8m impulse response
                elseif mic_position_type == 1
                    %convolve with Kayser IE anechoic 0.8m impulse response
                elseif mic_position_type == 2
                    %convolve with BKwHA_bte anechoic impulse response
                elseif mic_position_type == 3
                    %convolve with BKwHA_ie anechoic impulse response
                elseif mic_position_type == 4
                    %convolve with DADEC_ie anechoic impulse response
                elseif mic_position_type == 5
                    %convolve with Head_ie anechoic impulse response
                elseif mic_position_type == 6
                    %convolve with KEMAR anechoic impulse response
                end


            end
            %}
            %az_index_pool = 1:size(az_target,2);
            %el_index_pool = 1:size(el_target,2);
            %alpha_index_pool = 1:size(alpha,2);
            %room_dim_index_pool = 1:size(dimensions,1);
            %radius_index_pool = 1:size(radii_target,2);

            az_index = az_index_pool(randperm(size(az_index_pool,2),1));
            az_index_pool=az_index_pool(az_index_pool~=az_index);

            el_index = el_index_pool(randperm(size(el_index_pool,2),1));
            %el_index = nearest(rand*size(el_index_pool,2)+1);
            el_index_pool=el_index_pool(el_index_pool~=el_index);

            radius_ratio_index = radius_ratio_index_pool(randperm(size(radius_ratio_index_pool,2),1));
            radius_ratio_index_pool=radius_ratio_index_pool(radius_ratio_index_pool~=radius_ratio_index);

            alpha_index = alpha_index_pool(randperm(size(alpha_index_pool,2),1));
            %alpha_index = nearest(rand*size(alpha_index_pool,2)+1);
            alpha_index_pool=alpha_index_pool(alpha_index_pool~=alpha_index);

            alpha_add_index = alpha_add_index_pool(randperm(size(alpha_add_index_pool,2),1));
            %alpha_index = nearest(rand*size(alpha_index_pool,2)+1);
            alpha_add_index_pool=alpha_add_index_pool(alpha_add_index_pool~=alpha_add_index);

            x_room_dim_index = x_room_dim_index_pool(randperm(size(x_room_dim_index_pool,2),1));
            x_room_dim_index_pool=x_room_dim_index_pool(x_room_dim_index_pool~=x_room_dim_index);

            y_room_dim_index = y_room_dim_index_pool(randperm(size(y_room_dim_index_pool,2),1));
            y_room_dim_index_pool=y_room_dim_index_pool(y_room_dim_index_pool~=y_room_dim_index);

            z_room_dim_index = z_room_dim_index_pool(randperm(size(z_room_dim_index_pool,2),1));
            z_room_dim_index_pool=z_room_dim_index_pool(z_room_dim_index_pool~=z_room_dim_index);

            room_dimension = [dimension_x(x_room_dim_index),dimension_y(y_room_dim_index),dimension_z(z_room_dim_index)];
            
            if size(mic_position,1) > 2
                mic_position_temp(1,:) = (mic_position(1,:) + mic_position(3,:))/2;
                mic_position_temp(2,:) = (mic_position(2,:) + mic_position(4,:))/2;
            end
            
            alpha_vals = alpha_array(alpha_index,:) + alpha_add(alpha_add_index);

            if ~any(alpha_vals >= 1)
                
                [h, radius_target] = gen_spatial_IR_2021_04_10(mic_position,az_target(1,az_index), el_target(1,el_index), radii_target_ratio(radius_ratio_index), room_dimension, alpha_vals);

            
                if size(h,2) == 4
                    BRIR(:,1) = (h(:,1) + h(:,3))/2;
                    BRIR(:,2) = (h(:,2) + h(:,4))/2;
                elseif size(h,2) == 2
                    BRIR = h;
                else 
                    error('microphone dimensions must be 2 or 4 channel');
                end
            end
        else 
            error('Something went wrong with the choice of simulated or real impulse respone');
        end
        %this is where if I were to do data augmentation I might modify some things
        if measured_vs_simulated_index == 1
            speech_binaural = fftfilt(BRIR,anechoic_speech);
        end
        if measured_vs_simulated_index == 2 
            if ~any(alpha_vals >= 1)
                speech_binaural = fftfilt(BRIR,anechoic_speech);
            else
                audio_distance = abs(az_target(az_index)-180)/180;
                speech_binaural = [anechoic_speech*(1-audio_distance),anechoic_speech*audio_distance];
            end
        end
        %{
        noise_index = noise_index_pool(randperm(size(noise_index_pool,2),1));
        %noise_index = nearest(rand*size(noise_index_pool,1)+1);
        noise_index_pool=noise_index_pool(noise_index_pool~=noise_index);
        %do the noise stuff here
        noise_dir = strcat(base_dir,'\',noise_name_array{noise_index,1},'\',noise_name_array{noise_index,2},'.wav');
        %}
        noise_index = noise_index_pool(randperm(size(noise_index_pool,2),1));
        noise_index_pool=noise_index_pool(noise_index_pool~=noise_index);
        noise_dir = noise_wav_dirs{noise_index};
        [noise_wav,fs_noise] = audioread(noise_dir);
        if fs_noise ~= fs_speech
            noise_wav = resample(noise_wav,fs_speech,fs_noise);
            fs_noise = fs_speech;
        end
        noise_mono = zeros(size(noise_wav,1),1);

        for chan = 1:size(noise_wav,2)
            noise_mono = noise_mono + noise_wav(:,chan);
        end
        %noise_mono = noise_mono/max(max(abs(noise_mono)));
        if 2*size(anechoic_speech,1) > length(noise_mono)
            noise_mono_temp = [noise_mono;noise_mono];
            noise_mono = noise_mono_temp;
        end
        diffuse_noise = gen_isotropic_noise_2020_12_17(noise_mono,fs_noise,mic_position,size(anechoic_speech,1));
        if size(diffuse_noise,1) ~= size(speech_binaural,1)
            diffuse_noise = diffuse_noise(1:size(speech_binaural,1),1);
        end


        fs = fs_speech;








        s_db=v_activlev(speech_binaural,16000,'d');  % speech level in dB
        mod_binaural_speech = 10^((-(s_db))/20)*speech_binaural;
        s_db=v_activlev(mod_binaural_speech,16000,'d');  % speech level in dB
        mod_binaural_speech = 10^((-(s_db))/20)*mod_binaural_speech;
        s_db = v_activlev(mod_binaural_speech,16000,'d');  % speech level in dB
        n_db = mean(pow2db(sum(diffuse_noise.^2)/size(diffuse_noise,1)));

        SNR_diff_index = SNR_diff_index_pool(randperm(length(SNR_diff_index_pool),1));
        SNR_diff_index_pool=SNR_diff_index_pool(SNR_diff_index_pool~=SNR_diff_index);

        binaural_out = mod_binaural_speech + 10^(-(SNR_diff(SNR_diff_index) + n_db)/20)*diffuse_noise;


        direc_index = direc_pool(randperm(length(direc_pool),1));
        direc_pool =direc_pool(direc_pool~=direc_index);
        
        if direc_on(direc_index) == 1
            direc_type_index = direc_type_pool(randperm(length(direc_type_pool),1));
            direc_type_pool = direc_type_pool(direc_type_pool~=direc_type_index);
            
            SNR_direc_index = SNR_direc_index_pool(randperm(length(SNR_direc_index_pool),1));
            SNR_direc_index_pool=SNR_direc_index_pool(SNR_direc_index_pool~=SNR_direc_index);
            
            Kayser_strfind = strfind(BRIR_dir,'Kayser');
            AIR_strfind = strfind(BRIR_dir,'Aachen');
            eBrIRD_strfind = strfind(BRIR_dir,'eBrIRD');
            
            if ~isempty(eBrIRD_strfind)
                subdir_ind = strfind(BRIR_dir,'\');
                last_slash = subdir_ind(end);
                second_2last_slash = subdir_ind(end -1);
                direc_end = BRIR_dir(last_slash:end);
                cur_subdir = BRIR_dir(second_2last_slash + 1:last_slash - 1);
                dir_temp = BRIR_dir(1:second_2last_slash);
                dir_list=dir(dir_temp);
                dirflags = [dir_list.isdir];
                subdir = dir_list(dirflags);
                subdir(ismember( {subdir.name}, {'.', '..'})) = [];  %remove . and ..
                %nb_subdir = length(subdir);
                subdirs = {subdir.name};
                count = 1;
                subdirs(ismember(subdirs,cur_subdir)) = [];
                %{
                for i = 1:length(subdirs)
                    if ~strcmp(subdirs{i},cur_subdir)       
                        subdirs_temp{count} = subdirs{i};
                        count = count + 1;
                    end
                end
                %}
                %subdirs{subdirs == cur_subdir} = [];
                new_subdir = randsample(subdirs,1);
                new_subdir = new_subdir{1};
                direc_BRIR_dir = strcat(dir_temp,new_subdir,direc_end);
                [BRIR_direc_temp, fs_brir] = audioread(BRIR_dir);
                BRIR_direc_temp = resample(BRIR_direc_temp,fs,fs_brir);
                
                BRIR_direc_temp = BRIR_direc_temp(:,3:6);
                BRIR_direc(:,1) = (BRIR_direc_temp(:,1) + BRIR_direc_temp(:,3))/2; %left brir
                BRIR_direc(:,2) = (BRIR_direc_temp(:,2) + BRIR_direc_temp(:,4))/2; %right brir
            elseif ~isempty(AIR_strfind)
                subdir_ind = strfind(BRIR_dir,'\');
                last_slash = subdir_ind(end);
                direc_end = BRIR_dir(last_slash+1:end);
                dir_temp = BRIR_dir(1:last_slash);
                dir_list=dir(dir_temp);
                dir_list(ismember( {dir_list.name}, {'.', '..'})) = [];
                subdirs = {dir_list.name};
                %dirflags = [dir_list.isdir];
                %subdir = dir_list(dirflags);
                %subdir(ismember( {subdir.name}, {'.', '..'})) = [];
                %subdirs = {subdir.name};
                first_1_temp = strfind(BRIR_dir,'1');
                identifier = BRIR_dir(last_slash + 1:first_1_temp - 2);
                i= 1;
                while(i <= length(subdirs))
                    if ~contains(subdirs{i},identifier)
                        subdirs(i) = [];
                    else
                        i = i + 1;
                    end
                end
                %subdirs{contains(subdirs,identifier)} = [];
                subdirs(ismember(subdirs,direc_end)) = [];
                %{
                count = 1;
                for i = 1:length(subdirs)
                    if ~strcmp(subdirs{i},direc_end)      
                        subdirs_temp{count} = subdirs{i};
                        count = count + 1;
                    end
                end
                subdirs = subdirs_temp;
                %}
                %subdirs{subdirs == direc_end} = [];
                new_subdir = randsample(subdirs,1);
                new_subdir = new_subdir{1};
                direc_BRIR_dir = strcat(dir_temp,'\',new_subdir);
                [BRIR_direc, fs_brir] = audioread(BRIR_dir);
                
                BRIR_direc = resample(BRIR_direc,fs,fs_brir);
            elseif ~isempty(Kayser_strfind)
                subdir_ind = strfind(BRIR_dir,'\');
                last_slash = subdir_ind(end);
                direc_end = BRIR_dir(last_slash+1:end);
                dir_temp = BRIR_dir(1:last_slash);
                dir_list=dir(dir_temp);
                dir_list(ismember( {dir_list.name}, {'.', '..'})) = [];
                subdirs = {dir_list.name};
                
                subdirs(ismember(subdirs,direc_end)) = [];
                %dirflags = [dir_list.isdir];
                %subdir = dir_list(dirflags);

                %subdirs = {subdir.name};
                %{
                subdirs_temp = cell(length(subdirs),1);
                count = 1;
                for i = 1:length(subdirs)
                    if ~strcmp(subdirs{i},direc_end)      
                        subdirs_temp{count} = subdirs{i};
                        count = count + 1;
                    end
                end
                subdirs = subdirs_temp;
                %}
                %subdirs{subdirs == direc_end} = [];
                new_subdir = randsample(subdirs,1);
                new_subdir = new_subdir{1};
                direc_BRIR_dir = strcat(dir_temp,'\',new_subdir);
                [BRIR_direc_temp, fs_brir] = audioread(BRIR_dir);
                BRIR_direc_temp = resample(BRIR_direc_temp,fs,fs_brir);
                if contains(direc_BRIR_dir,'office_I')
                    BRIR_direc(:,1) = (HRTF(:,1) + HRTF(:,5))/2;
                    BRIR_direc(:,2) = (HRTF(:,2) + HRTF(:,6))/2;
                else
                    BRIR_direc(:,1) = (BRIR_direc_temp(:,3) + BRIR_direc_temp(:,7))/2;
                    BRIR_direc(:,2) = (BRIR_direc_temp(:,4) + BRIR_direc_temp(:,8))/2;
                end
            end
            
            if direc_type(direc_type_index) == 0
                direc_file_index = floor(rand(1)*size(noise_wav_dirs,1) + 1);
                [direc, fs_direc] = audioread(noise_wav_dirs{direc_file_index});
                if size(direc,2) > 1
                    direc = mean(direc,2);
                end
                if fs_direc ~= fs
                    direc = resample(direc,fs,fs_direc);
                end
                direc = direc(1:size(speech_binaural,1),1);
                
                direc_binaural = fftfilt(BRIR_direc,direc);

                nd_db = mean(pow2db(sum(diffuse_noise.^2)/size(diffuse_noise,1)));
                direc_binaural = 10^((-(nd_db) - SNR_direc(SNR_direc_index))/20)*direc_binaural;
                elseif direc_type(direc_type_index) == 1
                TIMIT_direc_dir = 'blabla';
                while(~contains(TIMIT_direc_dir,out_data_subdir))
                    TIMIT_direc_dir = randsample(TIMIT_wav_dirs,1);
                    TIMIT_direc_dir = TIMIT_direc_dir{1};
                end
                [direc,fs_direc] = audioread(TIMIT_direc_dir);
                if fs_direc ~= fs
                    direc = resample(direc,fs,fs_direc);
                end
                if length(direc) > size(speech_binaural,1)
                    direc = direc(1:size(speech_binaural,1),1);
                elseif length(direc) < size(speech_binaural,1)
                    direc = [direc;zeros(size(speech_binaural,1) - length(direc),1)];
                end
                direc_binaural = fftfilt(BRIR_direc,direc);
                sd_db = v_activlev(speech_binaural,16000,'d');
                direc_binaural = 10^((-(sd_db) - SNR_direc(SNR_direc_index))/20)*direc_binaural;
            end

            binaural_out = binaural_out + direc_binaural;
        end
            
                
                
                
                
                
                
                
                
            
            
                
                
                
            
            
        %{
        if direc_on(direc_index) == 1
            direc_type_index = direc_type_pool(randperm(length(direc_type_pool),1));
            direc_type_pool = direc_type_pool(direc_type_pool~=direc_type_index);

            SNR_direc_index = SNR_direc_index_pool(randperm(length(SNR_direc_index_pool),1));
            SNR_direc_index_pool=SNR_direc_index_pool(SNR_direc_index_pool~=SNR_direc_index);

            az_index_direc = floor(rand(1)*length(az_target)+ 1);
            el_index_direc = floor(rand(1)*length(el_target)+ 1);
            radius_target_ratio_direc = floor(rand(1)*length(radii_target_ratio)+ 1);
            if ~any(alpha_vals >= 1)
                [h_direc, radius_direc] = gen_spatial_IR_2021_04_10(mic_position,az_target(az_index_direc), el_target(el_index_direc),radii_target_ratio(radius_target_ratio_direc), room_dimension, alpha_vals);
            end
            if direc_type(direc_type_index) == 0
                direc_file_index = floor(rand(1)*size(noise_wav_dirs,1) + 1);
                [direc, fs_direc] = audioread(noise_wav_dirs{direc_file_index});
                if size(direc,2) > 1
                    direc = mean(direc,2);
                end
                if fs_direc ~= fs
                    direc = resample(direc,fs,fs_direc);
                end
                direc = direc(1:size(speech_binaural,1),1);
                if ~any(alpha_vals >= 1)
                    direc_binaural = fftfilt(h_direc,direc);
                else
                    audio_distance = abs(az_target(az_index_direc)-180)/180;
                    direc_binaural = [direc*(1-audio_distance),direc*audio_distance];
                end
                nd_db = mean(pow2db(sum(diffuse_noise.^2)/size(diffuse_noise,1)));
                direc_binaural = 10^((-(nd_db) - SNR_direc(SNR_direc_index))/20)*direc_binaural;
            elseif direc_type(direc_type_index) == 1
                TIMIT_direc_dir = ' ';
                while(~contains(TIMIT_direc_dir,out_data_subdir))
                    direc_file_index = floor(rand(1)*size(TIMIT_wav_dirs,1) + 1);
                    TIMIT_direc_dir = TIMIT_wav_dirs{direc_file_index};
                end
                [direc,fs_direc] = audioread(TIMIT_direc_dir);
                if fs_direc ~= fs
                    direc = resample(direc,fs,fs_direc);
                end
                if length(direc) > size(speech_binaural,1)
                    direc = direc(1:size(speech_binaural,1),1);
                elseif length(direc) < size(speech_binaural,1)
                    direc = [direc;zeros(size(speech_binaural,1) - length(direc),1)];
                end
                if ~any(alpha_vals >= 1)
                    direc_binaural = fftfilt(h_direc,direc);
                else
                    audio_distance = abs(az_target(az_index_direc)-180)/180;
                    direc_binaural = [direc*(1-audio_distance),direc*audio_distance];
                end
                sd_db = v_activlev(speech_binaural,16000,'d');
                direc_binaural = 10^((-(sd_db) - SNR_direc(SNR_direc_index))/20)*direc_binaural;
            end

            binaural_out = binaural_out + direc_binaural;
        end
        %}
        binaural_out = reshape(zscore(binaural_out(:)),size(binaural_out,1),size(binaural_out,2));
        anechoic_speech = reshape(zscore(anechoic_speech(:)),size(anechoic_speech,1),size(anechoic_speech,2));



        %xl = anechoic_speech_norm;
        %xr = anechoic_speech_norm;

        %yl = binaural_out_norm(:,1);
        %yr = binaural_out_norm(:,2);

        %mbstoi_norm = mbstoi_intermediate(xl,xr,yl,yr,fs_speech);

        xl = anechoic_speech;
        xr = anechoic_speech;

        yl = binaural_out(:,1);
        yr = binaural_out(:,2);

        mbstoi = mbstoi_intermediate(xl,xr,yl,yr,fs_speech);
        %{
        out_base_dir = 'E:\FYP\ML_data_dir\1-fold';
        %equal_check = isequal(mbstoi,mbstoi_norm);
        if measured_vs_simulated_index == 1
            out_base_dir = strcat(out_base_dir,'\measured');
        elseif measured_vs_simulated_index == 2
            out_base_dir = strcat(out_base_dir,'\simulated');
        end
        %}
        if ~isempty(TRAIN_strfind)
            out_dir = strcat(out_base_dir,'\',out_data_subdir,'\',num2str(TIMIT_train_file_count));
        elseif ~isempty(TEST_strfind)
            out_dir = strcat(out_base_dir,'\',out_data_subdir,'\',num2str(TIMIT_test_file_count));
        else
            error('something went wrong')
        end
        disp(out_dir);
        mkdir(out_dir);
        mat_file_name = strcat(out_dir,'\','mbstoi.mat');
        save(mat_file_name,'mbstoi');
        %save(mat_file_name,'mbstoi_norm','mbstoi','equal_check');

        out_clean_wav_dir = strcat(out_dir,'/','clean.wav');
        out_mixed_wav_dir = strcat(out_dir,'\','mixed.wav');
        v_writewav(binaural_out,16000,out_mixed_wav_dir,[],[],[],[]);
        v_writewav(anechoic_speech,16000,out_clean_wav_dir,[],[],[],[]);
        fileID = fopen(strcat(out_dir,'\','log.txt'),'w');
        fprintf(fileID,'mbstoi.mat contains the full mbstoi array, where the first dimension corresponds to the one-third octave bands and the second dimension correpsonds to the intermediate intelligbility indexes (blocks of 4096 samples at 10000 Khz of pure speech');
        fprintf(fileID,'The clean reference speech used was anechoic TIMIT speech \n');
        %fprintf(fileID,'For both of these index 1 and 2 are using anechoic and reverberant speech as clean respectively \n');
        fprintf(fileID,'\n\n');
        fprintf(fileID,strcat(strrep(strcat('directory of TIMIT file used: ',TIMIT_wav_dirs{TIMIT_wav_index,1}),'\','/'),' \n'));
        fprintf(fileID,'SNR diffuse noise: %.1f',SNR_diff(SNR_diff_index));
        fprintf(fileID,'\n\n');
        if measured_vs_simulated_index == 2
            if ~any(alpha_vals >= 1)
                fprintf(fileID,'Impulse Response Parameters: \n');
                fprintf(fileID,'Simulated \n');
                fprintf(fileID,'room dimensions [%.1f,%.1f,%.1f] [x,y,z] \n',room_dimension);
                if ~any(alpha_vals >= 1)
                    fprintf(fileID,'radius of target (m): %.2f \n',radius_target);
                end
                fprintf(fileID,'azimuth angle of target (degrees): %.2f \n',az_target(1,az_index));
                fprintf(fileID,'90 degrees azimuth is the direction facing the listener, with 180 degrees being pure left channel and 0 degrees being pure right channel \n');
            
                fprintf(fileID,'elevation angle of target (degrees): %.2f \n',el_target(1,el_index));
           
                fprintf(fileID,'reverberation parameter alpha: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f,] [alpha_{x1}, alpha_{x2}, alpha_{y1}, alpha_{y2}, alpha_{z1}, alpha_{z2}] \n',alpha_array(alpha_index,:) + alpha_add(1,alpha_add_index));
            else
                fprintf(fileID,'anechoic \n');
                fprintf(fileID,'azimuth angle of target (degrees): %.2f \n',az_target(1,az_index));
            end
        elseif measured_vs_simulated_index == 1
            fprintf(fileID,'Impulse Response Parameters: \n');
            fprintf(fileID,'Measured \n');
            fprintf(fileID,strcat(strrep(strcat('directory of measured impulse response used: ',measured_BRIR_wav_dirs{measured_BRIR_index,1}),'\','/'),' \n'));
        else 
            error('something went wrong');
        end
        if direc_on(direc_index) == 1
            fprintf(fileID,'Directional Noise Included \n');
            fprintf(fileID,'Directional Noise Parameters: \n');
            fprintf(fileID,'SNR directional noise: %.1f \n',SNR_direc(SNR_direc_index));
            if direc_mes == 0
                if ~any(alpha_vals >= 1)
                    fprintf(fileID,'azimuth angle of direc. target (degrees): %.2f \n',az_target(1,az_index_direc));
                    fprintf(fileID,'elevation angle of direc. target (degrees): %.2f \n',el_target(1,el_index_direc));
                    fprintf(fileID,'radius of direc. target (m): %.2f \n',radius_direc);
                else
                    fprintf(fileID,'anechoic \n');
                    fprintf(fileID,'azimuth angle of target (degrees): %.2f \n',az_target(1,az_index_direc));
                end
            else
                fprintf(fileID,'Measured \n');
                fprintf(fileID,strcat(strrep(strcat('directory of measured impulse response used: ',direc_BRIR_dir),'\','/'),' \n'));
            end
            if direc_type(direc_type_index) == 0
                fprintf(fileID,'Non-speech directional noise: \n');
                fprintf(fileID,strcat(strrep(strcat('directory of noise file used: ',noise_wav_dirs{direc_file_index}),'\','/'),' \n'));
            elseif direc_type(direc_type_index) == 1
                fprintf(fileID,'Speech directional noise: \n');
                fprintf(fileID,strcat(strrep(strcat('directory of speech file used: ',TIMIT_direc_dir),'\','/'),' \n'));
            end
        end




        fclose(fileID);
        %measured_BRIR_pool = [];
        if ~isempty(TRAIN_strfind)
            TIMIT_train_file_count = TIMIT_train_file_count + 1;
        end
        if ~isempty(TEST_strfind)
            TIMIT_test_file_count = TIMIT_test_file_count + 1;
        end


    end
end
    
    
