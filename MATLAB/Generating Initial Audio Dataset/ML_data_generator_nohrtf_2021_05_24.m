clear all;
addpath(genpath('D:\FYP\MATLAB\MBSTOI'));
addpath(genpath('D:\FYP\MATLAB\submodules'));
addpath(genpath('D:\FYP\MATLAB\other_dependencies'));
addpath(genpath('D:\FYP\MATLAB\RIR-Generator-master'));
addpath(genpath('D:\FYP\MATLAB\ANF-Generator-master'));


%{
[x, fs_x] = audioread('anechoic_speech.wav');
y = rand(size(x,1),1);

[z,p,fs_o] = v_addnoise(x,fs_x,10,'dxopEkn',y,fs_x);
%}

TIMIT_wav_dirs = load('TIMIT_wav_dirs_split').TIMIT_wav_dirs_train;

base_dir = 'D:\FYP\MATLAB';
HRTF_base_dir = 'D:\FYP\MATLAB\HRTF';

out_base_dir = 'D:\FYP\feature_preprocessing_input\directional_nohrtf_2021_06_02\TRAIN';

%out_base_unnorm_dir = 'E:\FYP\ML_data_dir\simulated\unnorm\1-fold';

noise_wav_dirs = load('TUT_wav_dirs').TUT_wav_dirs;

mic_pos_SCUT = [-0.0900,0,0;
                 0.0900,0,0];
             
mic_pos_BKwHA_bte_array = [-0.0800,-0.0136,  0.0374;
            0.0800,-0.0136,0.0374;
            -0.0800,-0.0285,0.0327;
            0.0800,-0.0285,0.0327];

mic_pos_BKwHA_bte = [(mic_pos_BKwHA_bte_array(1,:) + mic_pos_BKwHA_bte_array(3,:))/2;
                        (mic_pos_BKwHA_bte_array(2,:) + mic_pos_BKwHA_bte_array(4,:))/2];

mic_pos_ARI_bte = [-0.0900,0,0;
                    0.0900,0,0];

mic_pos_FF_KOELN = [-0.0875,0,0;
                    0.0875,0,0];

TIMIT_wav_pool = [];

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
mic_pos_type_pool = [];

direc_pool = [];
direc_type_pool = [];


TIMIT_train_file_count = 1;

nb_folds = 3;

dimension_x = 3:10;
dimension_y = 3:15;
dimension_z = 3:7;

alpha_plasterboard_ceilling = 0.0983;
alpha_wood_ceilling = 0.147;

alpha_plaster = 0.0283;
alpha_plywood = 0.2;

alpha_carpet = 0.182;
alpha_wood_f = 0.05;

alpha_add = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7];

mic_pos_type = 1:4;

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
     
rt60 = [];
%az_target = -180:5:175;
az_target = 0:5:355;
el_target = [-20,-15,-10,-5,0,5,10,15,20];
radii_target_ratio = 0.25:0.05:0.75;
SNR_diff = -9:3:30;
SNR_direc = -6:3:15;
direc_type = [0,1]; %0 -> noise source, 1 -> speech
direc_on = [0,1,2]; %0 -> no directional interfer, 1 -> directional interfer

radius_target_direc = zeros(1,2);
h_omni_direc = cell(1,2);
direc_type_index = cell(1,2);
SNR_direc_index = cell(1,2);
hrtf_dir_direc = cell(1,2);
hrtf_out_dir_direc = cell(1,2);
HRIR_direc = cell(1,2);
BRIR_direc = cell(1,2);
noise_direc_dir = cell(1,2);
TIMIT_direc_dir = cell(1,2);

file_counter = 1;
for fold = 1:nb_folds
    if isempty(TIMIT_wav_pool)
        TIMIT_wav_pool = 1:size(TIMIT_wav_dirs,1);
    end

    while(~isempty(TIMIT_wav_pool))
        %clear BRIR
        %clear BRIR_direc
        %{
        if isempty(measured_BRIR_pool)
            measured_BRIR_pool = 1:size(measured_BRIR_wav_dirs,1);
        end
        %}
        if isempty(noise_index_pool)
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
        %{
        if isempty(measured_vs_simulated_pool)
            measured_vs_simulated_pool = 1:2;
        end
        %}
        if isempty(direc_pool)
            direc_pool = 1:3;
        end
        if isempty(direc_type_pool)
            direc_type_pool = 1:2;
        end
        if isempty(mic_pos_type_pool)
            mic_pos_type_pool = 1:length(mic_pos_type);
        end


        TIMIT_wav_index = TIMIT_wav_pool(randperm(size(TIMIT_wav_pool,2),1));
        TIMIT_wav_pool=TIMIT_wav_pool(TIMIT_wav_pool~=TIMIT_wav_index);
        anechoic_speech_dir = TIMIT_wav_dirs{TIMIT_wav_index,1};

        [anechoic_speech, fs_speech] = audioread(anechoic_speech_dir);
        %{
        if ~isempty(measured_BRIR_pool)
            measured_vs_simulated_index = measured_vs_simulated_pool(randperm(size(measured_vs_simulated_pool,2),1));
            measured_vs_simulated_pool=measured_vs_simulated_pool(measured_vs_simulated_pool~=measured_vs_simulated_index);
        else
            measured_vs_simulated_index = 2;
        end
        %}
        %{
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

        %}
        %simulated
        %mic_position_type = nearest(rand(6));

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

        alpha_vals = alpha_array(alpha_index,:) + alpha_add(alpha_add_index);
        
        mic_pos_type_index = mic_pos_type_pool(randperm(size(mic_pos_type_pool,2),1));
        mic_pos_type_pool=mic_pos_type_pool(mic_pos_type_pool~=mic_pos_type_index);
        mic_pos_choice = mic_pos_type(mic_pos_type_index);
        switch mic_pos_choice
            case 1
                mic_position = mic_pos_SCUT;
            case 2
                mic_position = mic_pos_ARI_bte;
            case 3
                mic_position = mic_pos_BKwHA_bte;
            case 4
                mic_position = mic_pos_FF_KOELN;
        end

        if ~any(alpha_vals >= 1)
            az_target_rir_gen = az_target(az_index) + 90;
            if az_target_rir_gen > 180
                az_target_rir_gen = az_target_rir_gen - 360;
            end
            [h_omni, radius_target] = gen_spatial_IR_2021_04_10(mic_position, az_target_rir_gen, el_target(1,el_index), radii_target_ratio(radius_ratio_index), room_dimension, alpha_vals);

            %the azimuth here will have to be modified to fit the
            %specifications of the hrtf databases
                   
            RIR = h_omni;

            %read hrtf based on given azimuth, elevation and radius
            %convolve single channel reverberant impulse response with
            %hrtf

        end
        if ~any(alpha_vals >= 1)
            speech_binaural = fftfilt(RIR,anechoic_speech);
        else
            audio_distance = abs(az_target(az_index)-90)/180;
            if audio_distance > 1
                audio_distance = abs(2-audio_distance);
            end
            speech_binaural = [anechoic_speech*(1-audio_distance),anechoic_speech*audio_distance];
        end
        %speech_binaural = fftfilt(BRIR, anechoic_speech);

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
        
        if direc_on(direc_index) == 1 || direc_on(direc_index) == 2
            direc_type_index{1} = direc_type_pool(randperm(length(direc_type_pool),1));
            direc_type_pool = direc_type_pool(direc_type_pool~=direc_type_index{1});
            
            SNR_direc_index{1} = SNR_direc_index_pool(randperm(length(SNR_direc_index_pool),1));
            
            SNR_direc_index_pool=SNR_direc_index_pool(SNR_direc_index_pool~=SNR_direc_index{1});
            
            SNR_direc_index{2} = randi(length(SNR_direc));




            if az_index > 1
                mod_az_index_pool = 1:(az_index - 1);
                if az_index < length(az_target)
                    mod_az_index_pool = [mod_az_index_pool, (az_index + 1):length(az_target)];
                end
            else
                mod_az_index_pool = (az_index + 1):length(az_target);
            end
            az_index_direc = randperm(length(mod_az_index_pool),2);


            
            if ~any(alpha_vals >= 1)
                


                el_index_direc = randperm(length(el_target),2);
                
                az_target_rir_gen_direc(1) = az_target(az_index_direc(1)) + 90;
                if az_target_rir_gen_direc(1) > 180
                    az_target_rir_gen_direc(1) = az_target_rir_gen_direc(1) - 360;
                end
                
                if radius_ratio_index > 1
                    mod_rr_index_pool = 1:(radius_ratio_index - 1);
                    if radius_ratio_index < length(radii_target_ratio)
                        mod_rr_index_pool = [mod_rr_index_pool, (radius_ratio_index + 1):length(radii_target_ratio)];
                    end
                else
                    mod_rr_index_pool = (radius_ratio_index + 1):length(radii_target_ratio);
                end
                radius_target_direc_ratio_index = randperm(length(mod_rr_index_pool),2);
                
                %random elevation angle
                %random azimuth angle that isn't original azimuth angle
                %random radius target ratio that isn't original radius
                %target ratio

                
                [h_omni_direc{1}, radius_target_direc(1)] = gen_spatial_IR_2021_04_10(mic_position,az_target_rir_gen_direc(1), el_target(el_index_direc(1)), radii_target_ratio(radius_target_direc_ratio_index(1)), room_dimension, alpha_vals);

                RIR_direc{1} = h_omni_direc{1};
                if direc_on(direc_index) == 2
                    direc_type_index{2} = randi(2);
                    %{
                    if isempty(az_index_pool)
                        az_index_pool = 1:length(az_target);
                    end
                    %}
                    az_target_rir_gen_direc(2) = az_target(az_index_direc(2)) + 90;
                    if az_target_rir_gen_direc(2) > 180
                        az_target_rir_gen_direc(2) = az_target_rir_gen_direc(2) - 360;
                    end
                    
                    
                    [h_omni_direc{2}, radius_target_direc(2)] = gen_spatial_IR_2021_04_10(mic_position, az_target_rir_gen_direc(2), el_target(el_index_direc(2)), radii_target_ratio(radius_target_direc_ratio_index(2)), room_dimension, alpha_vals);

                    RIR_direc{2} = h_omni_direc{2};
                end
                
            end
                
            %now in all events I have two BRIR_direc
            %here now I need to pull either one or two wav files (noise or
            %speech) and then perform the required convolutions
            %after that I can add these normally
            if direc_type(direc_type_index{1}) == 0
                direc_file_index = randi(size(noise_wav_dirs,1));
                noise_direc_dir{1} = noise_wav_dirs{direc_file_index};
                [direc_full, fs_direc] = audioread(noise_direc_dir{1});
                if size(direc_full,2) > 1
                    direc_full = mean(direc_full,2);
                end
                if fs_direc ~= fs
                    direc_full = resample(direc_full,fs,fs_direc);
                end
                [~,I] = max(direc_full);
                overshoot = round(size(speech_binaural,1)/2);
                out_length = size(speech_binaural,1);
                if I >= out_length/2
                    while overshoot + I > length(direc_full) || I - (out_length - overshoot) + 1 < 1
                        overshoot = round(overshoot/2);
                        if overshoot < 5
                            overshoot = 0;
                        end
                    end
                    direc = direc_full((I+overshoot - out_length + 1):(I + overshoot));

                elseif I <= out_length/2
                    while I - overshoot < 1 || I + (out_length - overshoot -1) > length(direc_full)
                        overshoot = round(overshoot/2);
                        if overshoot < 5
                            overshoot = 0;
                        end
                    end
                    direc = direc_full((I - overshoot):(I-overshoot + out_length - 1));
                end
                if ~any(alpha_vals >= 1)
                    direc_binaural{1} = fftfilt(RIR_direc{1},direc);
                else
                    
                    audio_distance = abs(az_target(az_index_direc(1))-90)/180;
                    if audio_distance > 1
                        audio_distance = abs(2-audio_distance);
                    end
                    direc_binaural{1} = [direc*(1-audio_distance),direc*audio_distance];
                end
                if any(isnan(direc_binaural{1}))
                    error('nan issue with direc_binaural 1 before SNR - noise')
                end
                direc_binaural_1_prev_1 = direc_binaural{1};
                nd_db = pow2db(mean(sum(direc_binaural{1}.^2)/size(direc_binaural{1},1)));
                %if sign(nd_db) .* isinf(nd_db) == 
                direc_binaural{1} = 10^((-(nd_db) - SNR_direc(SNR_direc_index{1}))/20)*direc_binaural{1};
                if any(isnan(direc_binaural{1}))
                    error('nan issue with direc_binaural 1 after SNR - noise')
                end
            elseif direc_type(direc_type_index{1}) == 1
                TIMIT_direc_dir{1} = randsample(TIMIT_wav_dirs,1);
                [direc,fs_direc] = audioread(TIMIT_direc_dir{1}{1});
                if fs_direc ~= fs
                    direc = resample(direc,fs,fs_direc);
                end
                if length(direc) > size(speech_binaural,1)
                    direc = direc(1:size(speech_binaural,1),1);
                elseif length(direc) < size(speech_binaural,1)
                    direc = [direc;zeros(size(speech_binaural,1) - length(direc),1)];
                end
                if ~any(alpha_vals >= 1)
                    direc_binaural{1} = fftfilt(RIR_direc{1},direc);
                else
                    audio_distance = abs(az_target(az_index_direc(1))-90)/180;
                    if audio_distance > 1
                        audio_distance = abs(2-audio_distance);
                    end
                    direc_binaural{1} = [direc*(1-audio_distance),direc*audio_distance];
                end
                direc_binaural_1_prev_2 = direc_binaural{1};
                if any(isnan(direc_binaural{1}))
                    error('nan issue with direc_binaural 1 before SNR - speech')
                end
                sd_db = v_activlev(direc_binaural{1},16000,'d');
                direc_binaural{1} = 10^((-(sd_db) - SNR_direc(SNR_direc_index{1}))/20)*direc_binaural{1};
                if any(isnan(direc_binaural{1}))
                    error('nan issue with direc_binaural 1 after SNR - speech')
                end
            end


            binaural_out = binaural_out + direc_binaural{1};
        end
        
        if direc_on(direc_index) == 2
            direc_type_index{2} = randi(2);

            if direc_type(direc_type_index{2}) == 0
                direc_file_index = randi(size(noise_wav_dirs,1));
                noise_direc_dir{2} = noise_wav_dirs{direc_file_index};
                [direc_full, fs_direc] = audioread(noise_direc_dir{2});
                if size(direc_full,2) > 1
                    direc_full = mean(direc_full,2);
                end
                if fs_direc ~= fs
                    direc_full = resample(direc_full,fs,fs_direc);
                end
                [~,I] = max(direc_full);
                overshoot = round(size(speech_binaural,1)/2);
                out_length = size(speech_binaural,1);
                if I >= out_length/2
                    while overshoot + I > length(direc_full) || I - (out_length - overshoot) + 1 < 1
                        overshoot = round(overshoot/2);
                        if overshoot < 5
                            overshoot = 0;
                        end
                    end
                    direc = direc_full((I+overshoot - out_length + 1):(I + overshoot));

                elseif I <= out_length/2
                    while I - overshoot < 1 || I + (out_length - overshoot -1) > length(direc_full)
                        overshoot = round(overshoot/2);
                        if overshoot < 5
                            overshoot = 0;
                        end
                    end
                    direc = direc_full((I - overshoot):(I-overshoot + out_length - 1));
                end
                if ~any(alpha_vals >= 1)
                    direc_binaural{2} = fftfilt(RIR_direc{2},direc);
                else
                    
                    audio_distance = abs(az_target(az_index_direc(2))-90)/180;
                    if audio_distance > 1
                        audio_distance = abs(2-audio_distance);
                    end
                    direc_binaural{2} = [direc*(1-audio_distance),direc*audio_distance];
                end
                if any(isnan(direc_binaural{2}))
                    error('nan issue with direc_binaural 2 before SNR - noise')
                end
                direc_binaural_2_prev_1 = direc_binaural{2};
                nd_db = pow2db(mean(sum(direc_binaural{2}.^2)/size(direc_binaural{2},1)));
                direc_binaural{2} = 10^((-(nd_db) - SNR_direc(SNR_direc_index{2}))/20)*direc_binaural{2};
                if any(isnan(direc_binaural{2}))
                    error('nan issue with direc_binaural 2 after SNR - noise')
                end
            elseif direc_type(direc_type_index{2}) == 1
                TIMIT_direc_dir{2} = randsample(TIMIT_wav_dirs,1);
                [direc,fs_direc] = audioread(TIMIT_direc_dir{2}{1});
                if fs_direc ~= fs
                    direc = resample(direc,fs,fs_direc);
                end
                if length(direc) > size(speech_binaural,1)
                    direc = direc(1:size(speech_binaural,1),1);
                elseif length(direc) < size(speech_binaural,1)
                    direc = [direc;zeros(size(speech_binaural,1) - length(direc),1)];
                end
                
                if ~any(alpha_vals >= 1)
                    direc_binaural{2} = fftfilt(RIR_direc{2},direc);
                else
                    
                    audio_distance = abs(az_target(az_index_direc(2))-90)/180;
                    if audio_distance > 1
                        audio_distance = abs(2-audio_distance);
                    end
                    direc_binaural{2} = [direc*(1-audio_distance),direc*audio_distance];
                end
                if any(isnan(direc_binaural{2}))
                    error('nan issue with direc_binaural 2 before SNR - speech')
                end
                direc_binaural_2_prev_2 = direc_binaural{2};
                sd_db = v_activlev(direc_binaural{2},16000,'d');
                direc_binaural{2} = 10^((-(sd_db) - SNR_direc(SNR_direc_index{2}))/20)*direc_binaural{2};
                if any(isnan(direc_binaural{2}))
                    error('nan issue with direc_binaural 2 after SNR - speech')
                end
            end

            binaural_out = binaural_out + direc_binaural{2};
        end
        if all(binaural_out == 0)
            error('bug with binaural out');
        end
        if any(isnan(binaural_out))
            error('bug with binaural out nan');
        end

            
            
            
            
            
        %{
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
        %}
            
        %here also have anechoic_hrtf_speech, where the speech has been
        %convolved with hrtf for mbstoi calculation
        
        %I have to make sure at this point that I have an hrtf saved in all
        %cases
        
        %anechoic_speech_hrtf = fftfilt(HRIR,anechoic_speech);
        
        
        binaural_out = reshape(zscore(binaural_out(:)),size(binaural_out,1),size(binaural_out,2));
        anechoic_speech = reshape(zscore(anechoic_speech(:)),size(anechoic_speech,1),size(anechoic_speech,2));
        %anechoic_speech_hrtf = reshape(zscore(anechoic_speech_hrtf(:)),size(anechoic_speech_hrtf,1),size(anechoic_speech_hrtf,2));
            
        
        xl = anechoic_speech;
        xr = anechoic_speech;
        %xl_hrtf = anechoic_speech_hrtf(:,1);
        %xr_hrtf = anechoic_speech_hrtf(:,2);

        yl = binaural_out(:,1);
        yr = binaural_out(:,2);
        %calculate both mbstoi for clean and hrtf anechoic speech
        mbstoi = mbstoi_intermediate(xl,xr,yl,yr,fs_speech);
        %mbstoi_hrtf = mbstoi_intermediate(xl_hrtf, xr_hrtf, yl, yr, fs_speech);
        if any(isnan(mbstoi))
            error('bug with mbstoi nan');
        end
        out_dir = strcat(out_base_dir,'\',num2str(file_counter));
        disp(out_dir);
        mkdir(out_dir);
        mbstoi_file_name = strcat(out_dir,'\','mbstoi.mat');
        %mbstoi_hrtf_file_name = strcat(out_dir,'\','mbstoi_hrtf.mat');
        save(mbstoi_file_name,'mbstoi');
        %save(mbstoi_hrtf_file_name, 'mbstoi_hrtf');
        %save(mat_file_name,'mbstoi_norm','mbstoi','equal_check');

        out_clean_wav_dir = strcat(out_dir,'\','clean.wav');
        %out_clean_hrtf_wav_dir = strcat(out_dir,'\','clean_hrtf.wav');
        out_mixed_wav_dir = strcat(out_dir,'\','mixed.wav');
        v_writewav(binaural_out,16000,out_mixed_wav_dir,[],[],[],[]);
        v_writewav(anechoic_speech,16000,out_clean_wav_dir,[],[],[],[]);
        %v_writewav(anechoic_speech_hrtf,16000,out_clean_hrtf_wav_dir,[],[],[],[]);
        fileID = fopen(strcat(out_dir,'\','log.txt'),'w');
        fprintf(fileID,'mbstoi.mat contains the full mbstoi array, where the first dimension corresponds to the one-third octave bands and the second dimension correpsonds to the intermediate intelligbility indexes (blocks of 4096 samples at 10000 Khz of pure speech');
        fprintf(fileID,'The clean reference speech used was anechoic TIMIT speech \n');
        
        %fprintf(fileID,'mbstoi_hrtf.mat contains the full mbstoi array, using the clean_hrf speech as the clean reference speech, where the first dimension corresponds to the one-third octave bands and the second dimension correpsonds to the intermediate intelligbility indexes (blocks of 4096 samples at 10000 Khz of pure speech');
        %fprintf(fileID,'The clean_hrtf reference speech is anechoic TIMIT speech convolved with the HRIR described below\n');
        %fprintf(fileID,'For both of these index 1 and 2 are using anechoic and reverberant speech as clean respectively \n');
        fprintf(fileID,'\n\n');
        fprintf(fileID,strcat(strrep(strcat('directory of TIMIT file used: ',TIMIT_wav_dirs{TIMIT_wav_index,1}),'\','/'),' \n'));
        fprintf(fileID,'SNR diffuse noise: %.1f',SNR_diff(SNR_diff_index));
        fprintf(fileID,'\n\n');
        if ~any(alpha_vals >= 1)
            %here hrtf used will have to be specified, ie its directory
            fprintf(fileID,'Impulse Response Parameters: \n');
            fprintf(fileID,'Simulated \n');
            fprintf(fileID,'room dimensions [%.1f,%.1f,%.1f] [x,y,z] \n',room_dimension);
            if ~any(alpha_vals >= 1)
                fprintf(fileID,'radius of target (m): %.2f \n',radius_target);
            end
            fprintf(fileID,'azimuth angle of target (degrees): %.2f \n',az_target(az_index));
            fprintf(fileID,'0 degrees azimuth is the direction facing the listener, with 90 degrees being pure left channel and 270 degrees being pure right channel \n');

            fprintf(fileID,'elevation angle of target (degrees): %.2f \n',el_target(el_index));

            fprintf(fileID,'reverberation parameter alpha: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f,] [alpha_{x1}, alpha_{x2}, alpha_{y1}, alpha_{y2}, alpha_{z1}, alpha_{z2}] \n',alpha_array(alpha_index,:) + alpha_add(1,alpha_add_index));
            %fprintf(fileID,strcat(strrep(strcat('directory of HRTF file used: ',hrtf_out_dir),'\','/'),' \n'));
        else
            fprintf(fileID,'anechoic \n');
            fprintf(fileID,'azimuth angle of target (degrees): %.2f \n',az_target(1,az_index));
            %fprintf(fileID,'elevation angle of target (degrees): %.2f \n',el_target(1,el_index));
            %fprintf(fileID,strcat(strrep(strcat('directory of HRTF file used: ',hrtf_out_dir),'\','/'),' \n'));
            %here hrtf used will have to be specified, ie its directory
        end
        %{
        elseif measured_vs_simulated_index == 1
            fprintf(fileID,'Impulse Response Parameters: \n');
            fprintf(fileID,'Measured \n');
            fprintf(fileID,strcat(strrep(strcat('directory of measured impulse response used: ',measured_BRIR_wav_dirs{measured_BRIR_index,1}),'\','/'),' \n'));
        else 
            error('something went wrong');
        end
        %}
        if direc_on(direc_index) == 1 || direc_on(direc_index) == 2
            fprintf(fileID,'Directional Noise Included \n');
            if direc_on(direc_index) == 1
                fprintf(fileID,'Number of directional sources: 1 \n');
            else
                fprintf(fileID,'Number of directional sources: 2 \n');
            end
            fprintf(fileID,'First directional source: \n');
            fprintf(fileID,'SNR directional noise: %.1f \n',SNR_direc(SNR_direc_index{1}));
            if ~any(alpha_vals >= 1)
                fprintf(fileID,'reverberant \n');
                fprintf(fileID,'azimuth angle of direc. target (degrees): %.2f \n',az_target(az_index_direc(1)));
                fprintf(fileID,'elevation angle of direc. target (degrees): %.2f \n',el_target(el_index_direc(1)));
                fprintf(fileID,'radius of direc. target (m): %.2f \n',radius_target_direc(1));
                %fprintf(fileID,strcat(strrep(strcat('directory of hrtf used: ',hrtf_out_dir_direc{1}),'\','/'),' \n'));
                
            else
                fprintf(fileID,'anechoic \n');
                fprintf(fileID,'azimuth angle of target (degrees): %.2f \n',az_target(az_index_direc(1)));
                %fprintf(fileID,'elevation angle of direc. target (degrees): %.2f \n',el_target(el_index_direc(1)));
                %fprintf(fileID,strcat(strrep(strcat('directory of hrtf used: ',hrtf_out_dir_direc{1}),'\','/'),' \n'));
                
            end
            if direc_type(direc_type_index{1}) == 0
                fprintf(fileID,'Non-speech directional noise: \n');
                fprintf(fileID,strcat(strrep(strcat('directory of noise file used: ',noise_direc_dir{1}),'\','/'),' \n'));
            elseif direc_type(direc_type_index{1}) == 1
                fprintf(fileID,'Speech directional noise: \n');
                fprintf(fileID,strcat(strrep(strcat('directory of speech file used: ',TIMIT_direc_dir{1}{1}),'\','/'),' \n'));
            end
            if direc_on(direc_index) == 2
                fprintf(fileID,'Second directional source: \n');
                fprintf(fileID,'SNR directional noise: %.1f \n',SNR_direc(SNR_direc_index{2}));
                if ~any(alpha_vals >= 1)
                    fprintf(fileID,'reverberant \n');
                    fprintf(fileID,'azimuth angle of direc. target (degrees): %.2f \n',az_target(az_index_direc(2)));
                    fprintf(fileID,'elevation angle of direc. target (degrees): %.2f \n',el_target(el_index_direc(2)));
                    fprintf(fileID,'radius of direc. target (m): %.2f \n',radius_target_direc(2));
                    %fprintf(fileID,strcat(strrep(strcat('directory of hrtf used: ',hrtf_out_dir_direc{2}),'\','/'),' \n'));

                else
                    fprintf(fileID,'anechoic \n');
                    fprintf(fileID,'azimuth angle of target (degrees): %.2f \n',az_target(az_index_direc(2)));
                    %fprintf(fileID,'elevation angle of direc. target (degrees): %.2f \n',el_target(el_index_direc{2}));
                    %fprintf(fileID,strcat(strrep(strcat('directory of hrtf used: ',hrtf_out_dir_direc{2}),'\','/'),' \n'));

                end
                if direc_type(direc_type_index{2}) == 0
                    fprintf(fileID,'Non-speech directional noise: \n');
                    fprintf(fileID,strcat(strrep(strcat('directory of noise file used: ',noise_direc_dir{2}),'\','/'),' \n'));
                elseif direc_type(direc_type_index{2}) == 1
                    fprintf(fileID,'Speech directional noise: \n');
                    fprintf(fileID,strcat(strrep(strcat('directory of speech file used: ',TIMIT_direc_dir{2}{1}),'\','/'),' \n'));
                end
            end
        end
        %}



        fclose(fileID);
        %measured_BRIR_pool = [];
        %{
        if ~isempty(TRAIN_strfind)
            TIMIT_train_file_count = TIMIT_train_file_count + 1;
        end
        if ~isempty(TEST_strfind)
            TIMIT_test_file_count = TIMIT_test_file_count + 1;
        end
        %}
        file_counter = file_counter + 1;


    end
end
    
    
