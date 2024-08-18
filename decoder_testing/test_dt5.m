
RESULTSFOLDER = '/media/cyberspace007/T7/martin/decoder_testing/processed/day_9/rec4_4_states';
experimentFolder = '/media/cyberspace007/T7/martin/decoder_testing/dt5/UP2001_2023_12_04_Rehabilitation_day9_BrainGPT/wisci/2023_12_04_14_38_51 (rec4 (4 states model))/ABSD data'


EXPE = dir(experimentFolder);
cnt=1;
datafiles={};

for i=1:length(EXPE)
    expe=EXPE(i);
    if regexp(expe.name,'.dt5')
        datafiles{cnt} = expe.name;
        cnt=cnt+1;
    end
end

%% prepare data

[time,isUpdating,predictionWeight, ...
    yDesired,probaPredicted,threshold,yPredicted, raw, X, trigs, events] = loadDataM(experimentFolder, datafiles);
%save(fullfile(RESULTSFOLDER, "x.mat"), 'X', '-v7.3');
save(fullfile(RESULTSFOLDER, "all.mat"), 'time', 'isUpdating', 'yDesired', 'probaPredicted', 'threshold', 'yPredicted', 'X', 'trigs', 'events', '-v7.3');

disp("done");
%%
