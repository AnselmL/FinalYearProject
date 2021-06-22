%clear all;
addpath(genpath('/home/al5517/MBSTOI'));
addpath(genpath('/home/al5517/submodules'));
addpath(genpath('/home/al5517/other_dependencies'));
addpath(genpath('/home/al5517/RIR-Generator-master'));
addpath(genpath('/home/al5517/ANF-Generator-master'));
addpath(genpath('/home/al5517/INF-Generator-master'));
addpath(genpath('/home/al5517/Generating_Initial_Dataset'));

%{
[x, fs_x] = audioread('anechoic_speech.wav');
y = rand(size(x,1),1);

[z,p,fs_o] = v_addnoise(x,fs_x,10,'dxopEkn',y,fs_x);
%}

TIMIT_wav_dirs2 = load('/home/al5517/Generating_Initial_Dataset/TIMIT_wav_dirs_server.mat');

TIMIT_wav_dirs = TIMIT_wav_dirs2.TIMIT_wav_dirs_server;
%TIMIT_wav_dirs_test = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_test;
%TIMIT_wav_dirs_train = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_train;
%TIMIT_wav_dirs_val = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_val;

measured_BRIR_wav_dirs2 = load('/home/al5517/Generating_Initial_Dataset/measured_BRIR_wav_dirs_server.mat');
measured_BRIR_wav_dirs = measured_BRIR_wav_dirs2.BRIR_wav_dirs_server;

base_dir = '/home/al5517';
%RIR_dir = 'E:\FYP\Room_IR_Database';
%RIR_file_ext = '.wav';

out_base_dir = '/home/al5517/ML_data/simulated/1-fold';


%az_target_sm_rm = -180:12.5:170;

%az_noise = [90];
%el_noise = [0];

%az_el_target = [0,0]; %this can also be matrix if multiple target azimuths and elevations are desired
%az_el_noise = [90,0]; %this can be matrix with column 1 for azimuth angle and column 2 for elevation angle
%row matrix of all desired target radii, specified in meters

%radii_noise = [1,3,5]; %row matrix of all desired noise radii, specified in 
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

mic_pos_eBrIRD_ie = [-0.0350,0,0;
                    0.0350,0,0];
%{
mic_pos_eBrIRD_bte = [0.0800,-0.0140,0.0250;
                     -0.0800,-0.0140,0.0250;
                     0.0800,-0.0260,0.0250;
                     -0.0800,-0.0260,0.0250];
                 %}
mic_pos_eBrIRD_bte = [-0.0800,-0.0200,0.0250;
                     0.0800,-0.0200,0.0250];
                 
mic_pos_AIR = [-0.0850,0,0;
               0.0850,0,0];

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
SNR_index_pool = [];
az_index_pool = [];
el_index_pool = [];
alpha_index_pool = [];
room_dim_index_pool = [];
radius_index_pool = [];
measured_vs_simulated_pool = [];
anechoic_mic_position_pool = [];
Kayser_eBrIRD_IR_type_pool = [];

TIMIT_train_file_count = 1;
TIMIT_test_file_count = 1;

%TIMIT_train_pool = 1:size(TIMIT_wav_dirs_train,2);
%TIMIT_val_pool = 1:size(TIMIT_wav_dirs_val,2);
%TIMIT_test_pool = 1:size(TIMIT_wav_dirs_test,2);

%mic_pos = gen_mic_pos_Kayser_2020_12_17();

dimensions = [2.5,2.5,3; 4,4,3; 6,6,3; 7.5,7.5,3;10,10,4; 12.5,12.5,4; 15,15,4;10,20,4; 20,20,5; 25,25,10;10,30,4; 30,30,15]; % room dimensions specified with [x,y,z] coordinates
alpha = [0.6,0.4,0.2]; %absorption coefficients
rt60 = [];
az_target = -180:10:170;
el_target = [-10,10];
radii_target = [0.8,3];
radii_target_sm_rm = ones(1,1);
radii_target_sm_rm(1) = 0.8;
SNR = -30:3:30;
if isempty(TIMIT_wav_pool)
    TIMIT_wav_pool = 1:size(TIMIT_wav_dirs,1);
end
if isempty(measured_BRIR_pool)
    measured_BRIR_pool = 1:size(measured_BRIR_wav_dirs,1);
end
while(~isempty(TIMIT_wav_pool))
    %clear BRIR
    
%measured_BRIR_pool = 1:size(measured_BRIR_wav_dirs,1);
%TIMIT_cat = ["train","val","test"];

%this is where the while(TIMIT_train_pool or TIMIT_val_pool or
%TIMIT_test_pool)
%{
if isempty(TIMIT_cat_pool)
    TIMIT_cat_pool = 1:size(TIMIT_cat,2);
end
%}
    if isempty(noise_index_pool)
        noise_index_pool = 1:size(noise_name_array,1);
    end
    if isempty(SNR_index_pool)    
        SNR_index_pool = 1:size(SNR,2);
    end
    if isempty(az_index_pool)
        az_index_pool = 1:size(az_target,2);
    end
    if isempty(el_index_pool)
        el_index_pool = 1:size(el_target,2);
    end
    if isempty(alpha_index_pool)
        alpha_index_pool = 1:size(alpha,2);
    end
    if isempty(room_dim_index_pool)
        room_dim_index_pool = 1:size(dimensions,1);
    end
    if isempty(radius_index_pool)
        radius_index_pool = 1:size(radii_target,2);
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
    measured_vs_simulated_index = 2;
    %fetching speech impulse response
    if  measured_vs_simulated_index == 1
        measured_BRIR_index = measured_BRIR_pool(randperm(size(measured_BRIR_pool,2),1));
        %measured_BRIR_index = nearest(rand*size(measured_BRIR_pool,2)+1);
        measured_BRIR_pool=measured_BRIR_pool(measured_BRIR_pool~=measured_BRIR_index);
        BRIR_dir = measured_BRIR_wav_dirs{measured_BRIR_index,1};
        [HRTF, fs_brir] = audioread(BRIR_dir);
        
        BRIR = zeros(size(HRTF,1),2);
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
        %az_index = nearest(rand*size(az_index_pool,2)+1);
        az_index_pool=az_index_pool(az_index_pool~=az_index);

        el_index = el_index_pool(randperm(size(el_index_pool,2),1));
        %el_index = nearest(rand*size(el_index_pool,2)+1);
        el_index_pool=el_index_pool(el_index_pool~=el_index);

        alpha_index = alpha_index_pool(randperm(size(alpha_index_pool,2),1));
        %alpha_index = nearest(rand*size(alpha_index_pool,2)+1);
        alpha_index_pool=alpha_index_pool(alpha_index_pool~=alpha_index);
        
        room_dim_index = room_dim_index_pool(randperm(size(room_dim_index_pool,2),1));
        %room_dim_index = nearest(rand*size(room_dim_index_pool,1)+1);
        room_dim_index_pool=room_dim_index_pool(room_dim_index_pool~=room_dim_index);
        
        if room_dim_index < 4
            h = gen_spatial_IR_2020_12_17(mic_position,az_target(1,az_index), el_target(1,el_index), radii_target_sm_rm, dimensions(room_dim_index,:), alpha(1,alpha_index));
        else
            radius_index = radius_index_pool(randperm(size(radius_index_pool,2),1));
            radius_index_pool=radius_index_pool(radius_index_pool~=radius_index);
            h = gen_spatial_IR_2020_12_17(mic_position,az_target(1,az_index), el_target(1,el_index), radii_target(1,radius_index), dimensions(room_dim_index,:), alpha(1,alpha_index));
        end
        BRIR = zeros(size(h,1),2);
        if size(h,2) == 4
            BRIR(:,1) = (h(:,1) + h(:,3))/2;
            BRIR(:,2) = (h(:,2) + h(:,4))/2;
        elseif size(h,2) == 2
            BRIR = h;
        else 
            error('microphone dimensions must be 2 or 4 channel');
        end
    else 
        error('Something went wrong with the choice of simulated or real impulse respone');
    end
    %this is where if I were to do data augmentation I might modify some things
    speech_binaural = fftfilt(BRIR,anechoic_speech);

    noise_index = noise_index_pool(randperm(size(noise_index_pool,2),1));
    %noise_index = nearest(rand*size(noise_index_pool,1)+1);
    noise_index_pool=noise_index_pool(noise_index_pool~=noise_index);
    %do the noise stuff here
    noise_dir = strcat(base_dir,'/',noise_name_array{noise_index,1},'/',noise_name_array{noise_index,2},'.wav');
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

    SNR_index = nearest(rand*size(SNR_index_pool,2)+1);
    SNR_index_pool=SNR_index_pool(SNR_index_pool~=SNR_index);
    
    binaural_out = mod_binaural_speech + 10^(-(SNR(1,SNR_index) + n_db)/20)*diffuse_noise;
    
    binaural_out = reshape(zscore(binaural_out(:)),size(binaural_out,1),size(binaural_out,2));
    anechoic_speech = reshape(zscore(anechoic_speech(:)),size(anechoic_speech,1),size(anechoic_speech,2));

    xl = anechoic_speech;
    xr = anechoic_speech;

    yl = binaural_out(:,1);
    yr = binaural_out(:,2);

    mbstoi = mbstoi_intermediate(xl,xr,yl,yr,fs_speech);
    if ~isempty(TRAIN_strfind)
        out_dir = strcat(out_base_dir,'/',out_data_subdir,'/',num2str(TIMIT_train_file_count));
    elseif ~isempty(TEST_strfind)
        out_dir = strcat(out_base_dir,'/',out_data_subdir,'/',num2str(TIMIT_test_file_count));
    else
        error('something went wrong')
    end
    mkdir(out_dir);
    mat_file_name = strcat(out_dir,'/','mbstoi.mat');
    save(mat_file_name,'mbstoi');
    out_mixed_wav_dir = strcat(out_dir,'/','mixed.wav');
    out_clean_wav_dir = strcat(out_dir,'/','clean.wav');
    %out_rev_wav_dir = strcat(out_dir,'/','rev.wav');
    v_writewav(binaural_out,16000,out_mixed_wav_dir,[],[],[],[]);
    v_writewav(anechoic_speech,16000,out_clean_wav_dir,[],[],[],[]);
    fileID = fopen(strcat(out_dir,'/','log.txt'),'w');
    fprintf(fileID,'mbstoi.mat contains the full mbstoi array, where the first dimension corresponds to the one-third octave bands and the second dimension correpsonds to the intermediate intelligbility indexes (blocks of 4096 samples at 10000 Khz of pure speech');
    fprintf(fileID,'The clean reference speech used was anechoic TIMIT speech \n');
    %fprintf(fileID,'For both of these index 1 and 2 are using anechoic and reverberant speech as clean respectively \n');
    fprintf(fileID,'\n\n');
    fprintf(fileID,strcat(strrep(strcat('directory of TIMIT file used: ',TIMIT_wav_dirs{TIMIT_wav_index,1}),'/','/'),' \n'));
    fprintf(fileID,'SNR: %.1f',SNR(1,SNR_index));
    fprintf(fileID,'\n\n');
    if measured_vs_simulated_index == 2
        fprintf(fileID,'Impulse Response Parameters: \n');
        fprintf(fileID,'Simulated \n');
        fprintf(fileID,'room dimensions [%.1f,%.1f,%.1f] [x,y,z] \n',dimensions(room_dim_index,:));
        if room_dim_index < 4
            fprintf(fileID,'radius of target: %.2f \n',0.8);
        else
            fprintf(fileID,'radius of target: %.2f \n',radii_target(1,radius_index));
        end   
        fprintf(fileID,'azimuth angle of target: %.2f \n',az_target(1,az_index));
        fprintf(fileID,'90 degrees azimuth is the direction facing the listener, with 180 degrees being pure left channel and 0 degrees being pure right channel');
        fprintf(fileID,'elevation angle of target: %.2f \n',el_target(1,el_index));
        fprintf(fileID,'reverberation parameter alpha: %.2f \n',alpha(1,alpha_index));
    elseif measured_vs_simulated_index == 1
        fprintf(fileID,'Impulse Response Parameters: \n');
        fprintf(fileID,'Measured \n');
        fprintf(fileID,strcat('directory of measured impulse response used: ',measured_BRIR_wav_dirs{measured_BRIR_index,1}),' \n');
    else 
        error('something went wrong');
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
    
    
%%%do all the writing to file stuff bit


%at the end of here I have a BRIR and the speech in the format the I want
%it

%if Kayser or eBrIRD convolve with anechoic speech version always

%if simulated sometimes convolve with a mono version of the speech signal

%At this stage have binaural speech and clean speech

%now I need to generate the diffuse noise according to the microphone
%position

%then add the noise at the chosen snr

%calculate MBSTOI

%generate .mat output with all the necessary ingredients mentioned above

%prev_used_TIMIT_index = [TIMIT_index,prev_used_TIMIT_index];


%{

%->fetch impulse response
%kayser

noise_index = nearest(rand * size(noise_name_array,1));
if ismember(noise_index,prev_used_index)
    while ~ismbember(noise_index,prev_used_index)
        noise_index = nearest(rand);
    end
end
prev_used_index = [prev_used_index, noise_index];
index_check = ismember(index_test,prev_used_index);
if all(index_check)
    prev_used_index = [];
end

%sensor_pos = gen_mic_pos_Kayser_2020_12_17();                
%At this point I already need to have my noise index and my impulse response 
%generate microphone position according to the impulse response
%I should also consider an acceptable range of mic positions for the
%simulated cases, taken from MMHR-HRTF

%if Kayser, use kayser mic position(half the time BTE half the time in-ear), convolve anechoic speech with anechoic
%kayser impulse response half the time, and half the time leave TIMIT speech as is

%if eBrIRD, use eBrIRD mic position(half the time BTE half the time in-ear), convolve anechoic speech with anechoic
%eBrIRD impulse response half the time, and half the time leave TIMIT
%speech as is

%If AIR, use AIR mic position, leave TIMIT speech as is

%For simulated case, split between 5 mic positions (kayser + 4 MMHR-HRTF
%positions, or TIMIT as is with Kayser, for all of these half the time BTE,
%half the time in ear


%First thing I do is pull a speech file

%Then I pull an impulse response

%I check which characteristics the impulse response has, ie simulated or
%which database

%According to which database or simulated, I use the corresponding
%microphone positions

%For Kayser, eBrIRD, I have will have a counter correpsonding to whether
%the IE or BTE microphone impulse response and position is used (1-2). Also have
%a code section, which should be able to be commented out if desired, with
%a counter that if 0 leaves the anechoic speech as is if 1 it implements
%data augmentation based on the impulse response type (IE or BTE) (anechoic file
%should be stereo)

%For simulated, I will have a counter based for which microphone positions
%are used (1-5). I will also have a counter (0,1) based on if data augmentation is
%applied. If 1 data augmentation is applied according to which microphone
%position is used (again here for data augmentation case, anechoic file
%will be stereo)

%For AIR, I use the corresponding microphone position for the noise and
%leave the speech signal as is


%I should output a file with the following information
%simulated or measured
%If measured: url of impulse response
%If simulated: microphone position used
%If simulated: data augmentation, using stereo anechoic clean speech (yes or no)
%If simulated string characteristics (x_..y_..z_..r_..alpha_.. +anything
%else)
%
%If Kayser or eBrIRD: microphone type (IE or BTE) chosen
%If Kayser or eBrIRD: data augmentation, using stereo anechoic clean speech
%(yes or no)
%url of noise type
%MBSTOI vector (full, with octave bands and all intermediate values)








noise_index = 0;
%for i = 1:size(noise_name_array,1)
noise_dir = strcat(base_dir,'\',noise_name_array{noise_index,1},'\',noise_name_array{i,2},'.wav');
[y,fs] = audioread(noise_dir);
y = resample(y,16000,fs);
y_mono = zeros(size(y,1),1);
for chan = 1:size(y,2)
    y_mono = y_mono + y(:,chan);
end
y_mono = y_mono/max(max(abs(y_mono)));
fs = 16000;
full_noise_array(i,:,:) = gen_isotropic_noise_2020_12_17(y_mono,fs,sensor_pos);
%end
save('full_noise_array','full_noise_array');
full_noise_array = load('full_noise_array','full_noise_array').full_noise_array;
TIMIT_wav_dirs_train = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_train;
TIMIT_wav_dirs_val = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_val;
TIMIT_wav_dirs_test = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_test;

mode = ["train","val","test"];


SNR = -30:3:30;


%radii_total = unique(cat(2,radii_target,radii_noise));
%az_total = unique(cat(2,az_target,az_noise));
%el_total = unique(cat(2,el_target,el_noise));

dimensions = [2.5,2.5,3; 4,4,3; 6,6,3; 7.5,7.5,3;10,10,4; 12.5,12.5,4; 15,15,4;10,20,4; 20,20,5; 25,25,10;10,30,4; 30,30,15]; % room dimensions specified with [x,y,z] coordinates
alpha = [0.6,0.4,0.2]; %absorption coefficients
rt60 = [];


noise_index = ;
prev_used_noise_index = [];

prev_used_TIMIT_index = [];
SNR_index = ;
prev_used_SNR_index = [];
alpha_index = ;
prev_used_alpha_index = [];
room_size_index = ;
prev_used_room_size_index = [];
radius_index = ;
prev_used_radius_index = [];

TIMIT_index = nearest(rand*size(TIMIT_wav_dirs,1));
if ismember(TIMIT_index,prev_used_TIMIT_index)
    while ~ismbember(TIMIT_index,prev_used_TIMIT_index)
        TIMIT_index = nearest(rand*size(TIMIT_wav_dirs,1));
    end
end
prev_used_TIMIT_index = [TIMIT_index,prev_used_TIMIT_index];

anechoic_speech_dir = TIMIT_wav_dirs{TIMIT_index,1};

[anechoic_speech, fs_speech] = audioread(anechoic_speech_dir);


%->fetch impulse response
%kayser

noise_index = nearest(rand * size(noise_name_array,1));
if ismember(noise_index,prev_used_index)
    while ~ismbember(noise_index,prev_used_index)
        noise_index = nearest(rand);
    end
end
prev_used_index = [prev_used_index, noise_index];
index_check = ismember(index_test,prev_used_index);
if all(index_check)
    prev_used_index = [];
end

%}
